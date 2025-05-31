import os
import numpy as np
import re
import edlib
from tqdm import tqdm
import multiprocessing


# 并行处理的辅助函数
def _process_query_chunk_matching_worker(args):
    """
    处理单个查询序列块与参考序列的匹配。

    参数:
        args (tuple): 包含以下元素的元组:
            q_idx (int): 查询块的索引。
            query_s (str): 完整的查询序列。
            ref_s (str): 完整的参考序列。
            chunk_sz (int): 块大小。
            k (int): k-mer 的大小。
            kmer_thresh (int): k-mer 匹配的阈值。
            query_s_len (int): 查询序列的长度。
            ref_s_len (int): 参考序列的长度。

    返回:
        tuple: 包含两个列表的元组 (fwd_matches, rc_matches)。
               fwd_matches: 正向匹配结果列表，每个元素为 (q_idx, r_start, score)。
               rc_matches: 反向互补匹配结果列表，每个元素为 (q_idx, r_start, score)。
    """
    q_idx, query_s, ref_s, chunk_sz, k, kmer_thresh, query_s_len, ref_s_len = args

    q_s_offset = q_idx * chunk_sz
    q_e_offset = min((q_idx + 1) * chunk_sz, query_s_len)
    q_chunk = query_s[q_s_offset:q_e_offset]
    chunk_sz = q_e_offset - q_s_offset  # 实际查询块大小

    fwd_matches = []
    rc_matches = []

    if len(q_chunk) < k:
        return fwd_matches, rc_matches

    # 正向匹配
    for r_start in range(ref_s_len - chunk_sz + 1):
        ref_chunk = ref_s[r_start : r_start + chunk_sz]
        # 使用 edlib 计算编辑距离作为匹配分数
        edit_distance = edlib.align(q_chunk, ref_chunk)['editDistance']
        score = chunk_sz - edit_distance # 分数越高，匹配越好
        if score >= kmer_thresh:
            fwd_matches.append((q_idx, r_start, score))

    # 反向互补匹配
    rc_q_chunk = reverse_complement(q_chunk)

    for r_start in range(ref_s_len - chunk_sz + 1):
        ref_chunk = ref_s[r_start : r_start + chunk_sz]
        edit_distance = edlib.align(rc_q_chunk, ref_chunk)['editDistance']
        score = chunk_sz - edit_distance # 分数越高，匹配越好
        if score >= kmer_thresh:
            rc_matches.append((q_idx, r_start, score))
            
    if not(fwd_matches or rc_matches):
        # 如果没有找到任何匹配，打印提示信息
        print(f"查询块 {q_idx} (长度 {len(q_chunk)}) 在参考序列中未找到匹配。")
        print(f"序列坐标：{q_s_offset}，参考序列长度: {ref_s_len}bp。")
    
    return fwd_matches, rc_matches

# --- 用于 calculate_value 的辅助函数 ---
def rc(seq):
    """计算反向互补序列的简写函数。"""
    return reverse_complement(seq)

def get_points(tuples_str):
    """
    从表示坐标元组的字符串中提取数字点。
    例如，从 " ( 0, 300, 0, 300 ) , ( 300, 400, 400, 500 ) " 提取数字。
    """
    numbers = re.findall(r"\\d+", tuples_str)
    return [int(num) for num in numbers]

def calculate_distance(ref_s, query_s, r_s_1b, r_e_1b, q_s_1b, q_e_1b):
    """
    计算两个序列片段之间的编辑距离（考虑正向和反向互补）。
    坐标为1-based闭区间。
    """
    # 将1-based闭区间坐标转换为0-based开区间用于Python切片
    A = ref_s[r_s_1b - 1 : r_e_1b] 
    a = query_s[q_s_1b - 1 : q_e_1b]
    _a = reverse_complement(a) # 计算查询片段的反向互补序列
    # 返回正向和反向互补匹配中较小的编辑距离
    return min(edlib.align(A, a)['editDistance'], edlib.align(A, _a)['editDistance'])

def calculate_value(tuples_str, ref_s, query_s):  
    """
    根据给定的坐标元组字符串、参考序列和查询序列计算总分。
    分数计算考虑了编辑距离和对齐长度，并对重叠和片段长度不足进行惩罚。
    """
    try:
        points_data = np.array(get_points(tuples_str)) 
        
        if len(points_data) == 0: # 如果没有点，分数为0
            return 0
        if len(points_data) % 4 != 0: # 点的数量必须是4的倍数
            print(f"警告: 坐标点数量 ({len(points_data)}) 无效，不是4的倍数。返回0。")
            return 0

        total_edit_dist = 0 # 总编辑距离
        total_aligned_len = 0 # 总对齐长度
        prev_q_e = 0 # 前一个查询片段的0-based开区间结束位置
        
        points_arr = points_data.reshape((-1, 4)) # 将点数据重塑为 (N, 4) 的数组
        points_arr = points_arr[points_arr[:, 0].argsort()] # 按查询起始位置排序

        for idx, onetuple in enumerate(points_arr):
            # onetuple 是 (q_s, q_e, r_s, r_e)，均为0-based开区间
            q_s, q_e, r_s, r_e = map(int, onetuple)
            
            # 检查重叠：如果前一个查询片段的结束位置大于当前查询片段的起始位置，则存在重叠
            if idx > 0 and prev_q_e > q_s: 
                print(f"在索引 {idx - 1} 和 {idx} 之间的片段检测到重叠:")
                print(f"前一个查询结束位置 (0-based open): {prev_q_e}, 当前查询起始位置 (0-based open): {q_s}")
                return 0 # 重叠则分数为0
            
            prev_q_e = q_e # 更新前一个查询片段的结束位置

            current_edit_dist = 0
            current_aligned_len = 0

            # 确保片段非空
            if q_s < q_e and r_s < r_e:
                # 将0-based开区间转换为1-based闭区间以调用 calculate_distance
                q_s_1b = q_s + 1
                q_e_1b = q_e 
                r_s_1b = r_s + 1
                r_e_1b = r_e

                current_edit_dist = calculate_distance(ref_s, query_s, r_s_1b, r_e_1b, q_s_1b, q_e_1b)
                current_aligned_len = (q_e - q_s) # 0-based开区间的长度为 e - s
            
            total_edit_dist += current_edit_dist
            total_aligned_len += current_aligned_len
            if current_aligned_len > 0: # 只有当片段贡献了对齐长度时才应用惩罚
                total_aligned_len -= 30  # 每个片段的惩罚

        final_score = max(total_aligned_len - total_edit_dist, 0) # 最终分数不能为负
        return final_score
        
    except Exception as e:
        print(f"calculate_value 函数出错: {e}")
        return 0

# --- K-mer 相关函数 ---
def get_hashes(s, k):
    """计算序列中所有 k-mer 的滚动哈希值集合。"""
    hashes = set()
    n = len(s)
    if n < k:
        return hashes
    
    base = 256  # 哈希计算的基数
    mod = 1000000007  # 大素数模数，减少哈希冲突

    # 计算第一个 k-mer 的哈希值
    current_hash = 0
    for i in range(k):
        current_hash = (current_hash * base + ord(s[i])) % mod
    hashes.add(current_hash)

    # 预计算 base^(k-1) % mod 用于滚动哈希
    base_k_minus_1 = pow(base, k - 1, mod)

    # 使用滚动哈希计算后续 k-mer 的哈希值
    for i in range(1, n - k + 1):
        current_hash = (current_hash - ord(s[i - 1]) * base_k_minus_1) % mod
        current_hash = (current_hash * base + ord(s[i + k - 1])) % mod
        hashes.add(current_hash)
    return hashes

def count_common_kmers(seq1, seq2, k=9):
    """计算两个序列之间共同 k-mer 的数量。"""
    if not seq1 or not seq2 or len(seq1) < k or len(seq2) < k:
        return 0
    hashes1 = get_hashes(seq1, k)
    hashes2 = get_hashes(seq2, k)
    common_hashes = hashes1.intersection(hashes2)
    return len(common_hashes)

# --- 序列工具函数 ---
def reverse_complement(seq):
    """计算 DNA 序列的反向互补序列。"""
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'} # N表示未知碱基
    return "".join(complement.get(base, base) for base in reversed(seq.upper()))

# --- 处理匹配点以识别线段和孤立点 ---
def process_matches_for_visualization(raw_matches, match_type):
    """
    处理原始匹配列表，识别共线段和孤立点，用于可视化。

    参数:
        raw_matches (list): 原始匹配点列表，每个元素为 (q_idx, r_idx, score)。
        match_type (str): 匹配类型，"forward" 或 "rc"。

    返回:
        tuple: (segments, isolated_points_data)
               segments: 共线段列表，每个元素为 (q_start, r_start, q_end, r_end)。
               isolated_points_data: 孤立点列表，每个元素为 (q, r, score)。
    """
    if not raw_matches:
        return [], []
    
    # 将匹配点及其分数存储在字典中，方便查找
    points_with_scores = {(m[0], m[1]): m[2] for m in raw_matches}
    available_points = set(points_with_scores.keys()) # 所有可用匹配点
    
    segments = [] # 存储识别出的共线段
    isolated_points_data = [] # 存储识别出的孤立点
    
    # 按 q_idx, r_idx 排序，确保处理顺序一致
    sorted_unique_points = sorted(list(available_points), key=lambda p: (p[0], p[1]))
    
    visited_in_segment = set() # 记录已包含在某个段中的点

    for q_start, r_start in sorted_unique_points:
        if (q_start, r_start) in visited_in_segment: # 如果点已处理过，则跳过
            continue
        
        current_segment_path = [(q_start, r_start)] # 当前段的路径
        q_curr, r_curr = q_start, r_start

        # 尝试扩展当前段
        while True:
            if match_type == "forward": # 正向匹配，下一个点应该是 (q+1, r+1)
                q_next, r_next = q_curr + 1, r_curr + 1
            else: # 反向互补匹配，下一个点应该是 (q+1, r-1)
                q_next, r_next = q_curr + 1, r_curr - 1
            
            # 如果下一个点存在且未被访问过
            if (q_next, r_next) in available_points and (q_next, r_next) not in visited_in_segment:
                current_segment_path.append((q_next, r_next))
                q_curr, r_curr = q_next, r_next
            else:
                break # 无法扩展或下一点已被使用
        
        if len(current_segment_path) > 1: # 如果路径长度大于1，则认为是一个段
            seg_q_start_coord, seg_r_start_coord = current_segment_path[0]
            seg_q_end_coord, seg_r_end_coord = current_segment_path[-1]
            segments.append((seg_q_start_coord, seg_r_start_coord, seg_q_end_coord, seg_r_end_coord))
            for p_item in current_segment_path: # 将段中的所有点标记为已访问
                visited_in_segment.add(p_item)
                
    # 收集孤立点：任何未包含在段中的原始匹配点
    for q, r in sorted_unique_points:
        if (q, r) not in visited_in_segment:
            isolated_points_data.append((q, r, points_with_scores[(q,r)]))
            
    return segments, isolated_points_data

# --- SVG 可视化函数 ---
def generate_svg_grid(query_s_len, ref_s_len, 
                      fwd_segs, fwd_pts,
                      rc_segs, rc_pts,
                      chunk_sz,
                      num_q_chunks, num_r_chunks,
                      output_filename="matches_grid_visualization.svg",
                      svg_padding=70, cell_size=10, point_radius=3,
                      long_path=None):
    """
    生成匹配结果的二维网格 SVG 可视化图像。
    包含线段、孤立点和最长路径高亮。
    """
    grid_width = num_q_chunks * cell_size
    grid_height = num_r_chunks * cell_size

    # 处理序列或区块数量为零的情况
    if num_q_chunks == 0 or num_r_chunks == 0:
        grid_width = max(grid_width, cell_size) # 至少保证一个单元格的宽度
        grid_height = max(grid_height, cell_size) # 至少保证一个单元格的高度

    svg_width = grid_width + 2 * svg_padding
    svg_height = grid_height + 2 * svg_padding

    svg_content = f'<svg width="{svg_width}" height="{svg_height}" xmlns="http://www.w3.org/2000/svg" font-family="Arial, sans-serif">\\n'
    svg_content += '  <style>\\n'
    svg_content += '    .grid_line { stroke: #E0E0E0; stroke-width: 0.5; }\\n'
    svg_content += '    .fwd_match_node { fill: blue; fill-opacity: 0.7; }\\n' # 正向匹配点样式
    svg_content += '    .rc_match_node { fill: red; fill-opacity: 0.7; }\\n'   # 反向互补匹配点样式
    svg_content += '    .fwd_match_segment { stroke: blue; stroke-width: 2; stroke-opacity: 0.7; }\\n' # 正向匹配线段样式
    svg_content += '    .rc_match_segment { stroke: red; stroke-width: 2; stroke-opacity: 0.7; }\\n'   # 反向互补匹配线段样式
    svg_content += '    .longest_path_line { stroke: purple; stroke-width: 2.5; stroke-opacity: 0.9; }\\n' # 最长路径连线样式
    svg_content += '    .longest_path_node_highlight { stroke: gold; stroke-width: 2px; fill-opacity: 1 !important; }\\n' # 最长路径节点高亮样式
    svg_content += '    .axis_label { font-size: 10px; text-anchor: middle; }\\n' # 坐标轴标签样式
    svg_content += '    .title_label { font-size: 14px; text-anchor: middle; font-weight: bold; }\\n' # 标题样式
    svg_content += '    .no_match_text { font-size: 12px; text-anchor: middle; fill: #555; }\\n' # 无匹配时提示文本样式
    svg_content += '    circle:hover { stroke: black; stroke-width: 1px; }\\n' # 鼠标悬停效果
    svg_content += '  </style>\\n'

    # 图表标题
    svg_content += f'  <text x="{svg_width / 2}" y="{svg_padding / 2.5}" class="title_label">查询块 vs 参考块 匹配网格</text>\\n'

    # 绘制网格背景和边框
    svg_content += f'  <rect x="{svg_padding}" y="{svg_padding}" width="{grid_width}" height="{grid_height}" fill="none" stroke="#333" stroke-width="1"/>\\n'

    if num_q_chunks == 0 or num_r_chunks == 0:
        svg_content += f'  <text x="{svg_padding + grid_width / 2}" y="{svg_padding + grid_height / 2}" class="no_match_text">无数据显示 (查询或参考块数量为零)。</text>\\n'
    else:
        # 绘制网格线
        for i in range(1, num_q_chunks): # 内部垂直线
            x = svg_padding + i * cell_size
            svg_content += f'  <line x1="{x}" y1="{svg_padding}" x2="{x}" y2="{svg_padding + grid_height}" class="grid_line" />\\n'
        for j in range(1, num_r_chunks): # 内部水平线
            y = svg_padding + j * cell_size
            svg_content += f'  <line x1="{svg_padding}" y1="{y}" x2="{svg_padding + grid_width}" y2="{y}" class="grid_line" />\\n'
        
        # 绘制轴标签
        # X 轴标签 (查询块)
        label_step_x = max(1, num_q_chunks // 20) # 调整标签密度避免重叠
        for i in range(num_q_chunks):
            if i % label_step_x == 0:
                x = svg_padding + (i + 0.5) * cell_size
                svg_content += f'  <text x="{x}" y="{svg_padding + grid_height + 15}" class="axis_label">{i}</text>\\n'
        svg_content += f'  <text x="{svg_padding + grid_width / 2}" y="{svg_padding + grid_height + 35}" class="axis_label">查询块索引 (块大小: {chunk_sz}bp, 总块数: {num_q_chunks})</text>\\n'
        
        # Y 轴标签 (参考块)
        label_step_y = max(1, num_r_chunks // 20)
        for j in range(num_r_chunks):
            if j % label_step_y == 0:
                y_pos = svg_padding + (j + 0.5) * cell_size
                x_pos = svg_padding - 15
                svg_content += f'  <text x="{x_pos}" y="{y_pos}" class="axis_label" transform="rotate(-90 {x_pos},{y_pos})\">{j}</text>\\n'
        svg_content += f'  <text x="{svg_padding - 45}" y="{svg_padding + grid_height / 2}" class="axis_label" transform="rotate(-90 {svg_padding - 45},{svg_padding + grid_height / 2})\">参考块索引 (总块数: {num_r_chunks})</text>\\n'

        # 绘制匹配节点和线段
        # 正向匹配 (蓝色)
        for q_s, r_s, q_e, r_e in fwd_segs: # 绘制线段
            x1 = svg_padding + (q_s + 0.5) * cell_size
            y1 = svg_padding + (r_s + 0.5) * cell_size
            x2 = svg_padding + (q_e + 0.5) * cell_size
            y2 = svg_padding + (r_e + 0.5) * cell_size
            svg_content += f'  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="fwd_match_segment">\\n'
            svg_content += f'    <title>正向线段\\n查询块: {q_s}-{q_e}\\n参考块: {r_s}-{r_e}</title>\\n'
            svg_content += f'  </line>\\n'
        for q_idx, r_idx, score in fwd_pts: # 绘制孤立点
            center_x = svg_padding + (q_idx + 0.5) * cell_size
            center_y = svg_padding + (r_idx + 0.5) * cell_size
            svg_content += f'  <circle cx="{center_x}" cy="{center_y}" r="{point_radius}" class="fwd_match_node">\\n'
            svg_content += f'    <title>正向匹配 (孤立点)\\n查询块: {q_idx}\\n参考块: {r_idx}\\n分数: {score}</title>\\n'
            svg_content += f'  </circle>\\n'

        # 反向互补匹配 (红色)
        for q_s, r_s, q_e, r_e in rc_segs: # 绘制线段
            x1 = svg_padding + (q_s + 0.5) * cell_size
            y1 = svg_padding + (r_s + 0.5) * cell_size
            x2 = svg_padding + (q_e + 0.5) * cell_size
            y2 = svg_padding + (r_e + 0.5) * cell_size
            svg_content += f'  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="rc_match_segment">\\n'
            svg_content += f'    <title>反向互补线段\\n查询块: {q_s}-{q_e}\\n参考块: {r_s}-{r_e}</title>\\n'
            svg_content += f'  </line>\\n'
        for q_idx, r_idx, score in rc_pts: # 绘制孤立点
            center_x = svg_padding + (q_idx + 0.5) * cell_size
            center_y = svg_padding + (r_idx + 0.5) * cell_size
            svg_content += f'  <circle cx="{center_x}" cy="{center_y}" r="{point_radius}" class="rc_match_node">\\n'
            svg_content += f'    <title>反向互补匹配 (孤立点)\\n查询块: {q_idx}\\n参考块: {r_idx}\\n分数: {score}</title>\\n'
            svg_content += f'  </circle>\\n'

        # 高亮显示最长路径（如果提供）
        if long_path and len(long_path) > 0:
            if len(long_path) > 1: # 绘制连接线
                for p_idx in range(len(long_path) - 1):
                    curr_n = long_path[p_idx]
                    next_n = long_path[p_idx+1]
                    x1 = svg_padding + (curr_n['q'] + 0.5) * cell_size
                    y1 = svg_padding + (curr_n['r'] + 0.5) * cell_size
                    x2 = svg_padding + (next_n['q'] + 0.5) * cell_size
                    y2 = svg_padding + (next_n['r'] + 0.5) * cell_size
                    svg_content += f'  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="longest_path_line" />\\n'
            for n_data in long_path: # 高亮路径上的节点
                center_x = svg_padding + (n_data['q'] + 0.5) * cell_size
                center_y = svg_padding + (n_data['r'] + 0.5) * cell_size
                base_class = "fwd_match_node" if n_data['type'] == 'forward' else "rc_match_node"
                svg_content += f'  <circle cx="{center_x}" cy="{center_y}" r="{point_radius + 1}" class="{base_class} longest_path_node_highlight">\\n' # 高亮节点半径稍大
                svg_content += f'    <title>{n_data["type"].capitalize()} 匹配 (最长路径)\\n查询块: {n_data["q"]}\\n参考块: {n_data["r"]}\\n原始分数: {n_data["original_score"]}\\n路径分数: {n_data.get("path_score_at_node", "N/A"):.2f}</title>\\n'
                svg_content += f'  </circle>\\n'
    svg_content += '</svg>\\n'
    with open(output_filename, 'w') as f:
        f.write(svg_content)
    print(f"SVG 网格可视化已保存到 {output_filename}")

# --- 主要处理逻辑 ---
def find_and_visualize_matches(query_f, ref_f, svg_out_f,
                               k=7, chunk_sz=50, kmer_thresh=5,
                               switching_penalty=1000, collinear_jump_penalty=10,
                               allowed_chunk_gap=0):
    """
    读取查询序列和参考序列，查找匹配（正向和反向互补），
    计算最长对齐路径，并生成 SVG 可视化结果。

    参数:
        query_f (str): 查询序列文件路径。
        ref_f (str): 参考序列文件路径。
        svg_out_f (str): 输出 SVG 文件路径。
        k (int): k-mer 大小。
        chunk_sz (int): 用于比较的序列块大小。
        kmer_thresh (int): k-mer 匹配阈值。
        switching_penalty (int): 切换匹配类型（正向/反向互补）的惩罚。
        collinear_jump_penalty (int): 共线跳跃的惩罚。
        allowed_chunk_gap (int): 允许合并的共线段之间最大的查询块间隔数。
                                 0 表示必须严格相邻。
    """
    try:
        with open(query_f, 'r') as f_in:
            query_s = f_in.read().replace('\\\\n', '').strip().upper() 
        with open(ref_f, 'r') as f_in:
            ref_s = f_in.read().replace('\\\\n', '').strip().upper()
    except FileNotFoundError as e:
        print(f"错误: {e}。请确保输入文件存在。")
        return

    if not query_s or not ref_s:
        print("错误: 查询序列或参考序列为空。")
        return

    query_s_len = len(query_s)
    ref_s_len = len(ref_s)

    fwd_matches_raw = [] # 存储原始正向匹配
    rc_matches_raw = []  # 存储原始反向互补匹配

    num_q_chunks = (query_s_len + chunk_sz - 1) // chunk_sz # 查询序列的块数
    ref_chunks_svg = (ref_s_len + chunk_sz - 1) // chunk_sz # 参考序列的块数（用于SVG）

    print(f"正在处理查询序列 ({query_s_len}bp) 和参考序列 ({ref_s_len}bp)...")
    print(f"K-mer 大小: {k}, 块大小: {chunk_sz}, K-mer 阈值: {kmer_thresh}, 切换惩罚: {switching_penalty}, 共线跳跃惩罚: {collinear_jump_penalty}, 允许的块间隔: {allowed_chunk_gap}") 
    print("注意: 块匹配分数基于edlib编辑距离。")

    # 准备并行处理的任务参数
    tasks = []
    for i in range(num_q_chunks):
        tasks.append((i, query_s, ref_s, chunk_sz, k, kmer_thresh, query_s_len, ref_s_len))

    # 使用多进程并行处理查询块
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool_res = pool.imap_unordered(_process_query_chunk_matching_worker, tasks)
        # 使用 tqdm 显示进度条
        for f_match_list, rc_match_list in tqdm(pool_res, total=len(tasks), desc="并行处理查询块"):
            if f_match_list:
                fwd_matches_raw.extend(f_match_list)
            if rc_match_list:
                rc_matches_raw.extend(rc_match_list)
    
    print(f"找到 {len(fwd_matches_raw)} 个正向匹配和 {len(rc_matches_raw)} 个反向互补匹配 (基于阈值)。")

    # --- 最长路径计算 ---
    nodes = [] # 存储所有匹配节点信息
    # r_node_pos 是参考块的起始位置 (0-based)
    for q_node_idx, r_node_pos, score in fwd_matches_raw:
        nodes.append({'q': q_node_idx, 'r': r_node_pos, 'score': float(score), 'type': 'forward', 'original_score': float(score)})
    for q_node_idx, r_node_pos, score in rc_matches_raw:
        nodes.append({'q': q_node_idx, 'r': r_node_pos, 'score': float(score), 'type': 'rc', 'original_score': float(score)})

    # 按 q_idx, r_idx, type 排序 (拓扑排序)
    nodes.sort(key=lambda n: (n['q'], n['r'], n['type']))

    longest_path_nodes = [] # 存储最长路径上的节点
    max_overall_score = -float('inf') # 初始化最大总分
    calculated_value_score = 0 # 初始化 calculate_value 计算的分数

    if not nodes:
        print("没有找到匹配节点，无法计算最长路径。")
    else:
        N_nodes = len(nodes)
        # dp[i] 存储以 nodes[i] 结尾的路径的最大分数
        dp = [node['original_score'] for node in nodes] 
        predecessor = [-1] * N_nodes  # 用于路径回溯的前驱节点索引

        # 动态规划计算最长路径
        for i in tqdm(range(N_nodes), desc="计算最长路径 (DP)"):
            n_i = nodes[i] # 当前节点 i
            q_i_idx, r_i_pos, type_i = n_i['q'], n_i['r'], n_i['type']
            
            # 遍历所有之前的节点 j 作为节点 i 的潜在前驱
            for j in range(i):
                n_j = nodes[j] # 前驱节点 j
                q_j_idx, r_j_pos, type_j = n_j['q'], n_j['r'], n_j['type']

                # 有效转换要求查询块索引增加
                if q_i_idx > q_j_idx:
                    current_penalty = 0 # 当前转换的惩罚
                    # 将块索引转换为碱基对坐标
                    q_i_bp = q_i_idx * chunk_sz 
                    q_j_bp = q_j_idx * chunk_sz 
                    delta_q_bp = q_i_bp - q_j_bp # 查询序列上碱基对的差值

                    if type_i != type_j: # 如果匹配类型切换（例如，正向到反向互补）
                        current_penalty = switching_penalty
                    else: # 匹配类型相同
                        is_collinear = False # 是否共线
                        if type_i == 'forward': # 正向匹配
                            expected_r_i_pos = r_j_pos + delta_q_bp # 期望的参考序列起始位置
                            if r_i_pos == expected_r_i_pos:
                                is_collinear = True
                        else: # 反向互补匹配
                            expected_r_i_pos = r_j_pos - delta_q_bp # 期望的参考序列起始位置
                            if r_i_pos == expected_r_i_pos:
                                is_collinear = True
                        
                        is_q_adj = (q_i_idx == q_j_idx + 1) # 查询块是否相邻

                        if is_q_adj: # 如果查询块相邻
                            if not is_collinear: # 但不共线
                                current_penalty = switching_penalty # 视为类型切换
                        else: # 如果查询块不相邻 (q_i_idx > q_j_idx + 1)
                            if is_collinear: # 但共线
                                current_penalty = collinear_jump_penalty # 共线跳跃惩罚
                            else: # 既不相邻也不共线
                                current_penalty = switching_penalty # 视为类型切换
                
                    # 候选分数 = 路径到j的分数 + 当前节点i的原始分数 - 惩罚
                    potential_score = dp[j] + n_i['original_score'] - current_penalty
                    
                    if potential_score > dp[i]: # 如果找到更高分数的路径
                        dp[i] = potential_score
                        predecessor[i] = j # 更新前驱
        
        # 找到 DP 表中的最大分数及其对应的结束节点
        max_dp_s = -float('inf')
        max_s_idx = -1
        if N_nodes > 0:
            for dp_i in range(N_nodes):
                if dp[dp_i] > max_dp_s:
                    max_dp_s = dp[dp_i]
                    max_s_idx = dp_i
            max_overall_score = max_dp_s

            # 回溯路径
            if max_s_idx != -1:
                curr_idx = max_s_idx
                while curr_idx != -1:
                    p_node = nodes[curr_idx].copy()
                    p_node['path_score_at_node'] = dp[curr_idx] # 存储节点处的累积路径分数
                    longest_path_nodes.append(p_node)
                    curr_idx = predecessor[curr_idx]
                longest_path_nodes.reverse() # 反转得到正确的路径顺序

        print(f"最长对齐路径分数: {max_overall_score if N_nodes > 0 else 'N/A'}")
        if longest_path_nodes:
            merged_segments_output = [] # 存储合并后的片段坐标
            if not longest_path_nodes: 
                pass 
            else:
                curr_seg = None # 当前正在合并的片段
                for i in range(len(longest_path_nodes)):
                    node = longest_path_nodes[i]
                    q_idx_seg = node['q']
                    r_pos_seg = node['r'] # 参考块的0-based起始位置
                    node_type = node['type']

                    # 计算当前块的0-based开区间坐标
                    q_s_bp = q_idx_seg * chunk_sz
                    q_e_bp = min((q_idx_seg + 1) * chunk_sz, query_s_len)
                    r_s_bp = r_pos_seg 
                    r_e_bp = min(r_pos_seg + chunk_sz, ref_s_len) 

                    if curr_seg is None: # 如果是第一个节点，开始新的合并片段
                        curr_seg = {
                            'q_start_merged_0based': q_s_bp,
                            'q_end_merged_0based_open': q_e_bp,
                            'ref_s_merged_0based': r_s_bp, 
                            'ref_e_merged_0based_open': r_e_bp,
                            'type': node_type,
                            'last_q_idx': q_idx_seg,
                            'last_r_idx': r_pos_seg 
                        }
                    else: # 尝试与前一个节点合并
                        is_collinear_cont = False # 是否为共线延续
                        delta_q_c = q_idx_seg - curr_seg['last_q_idx'] # 查询块索引的差值
                        
                        # 检查是否可以合并：类型相同，且查询块间隔在允许范围内
                        if (node_type == curr_seg['type'] and
                            0 < delta_q_c <= (allowed_chunk_gap + 1)):
                            
                            exp_r_pos = -1 # 期望的参考块起始位置
                            if node_type == 'forward':
                                exp_r_pos = curr_seg['last_r_idx'] + delta_q_c * chunk_sz
                            elif node_type == 'rc':
                                exp_r_pos = curr_seg['last_r_idx'] - delta_q_c * chunk_sz
                            
                            if r_pos_seg == exp_r_pos: # 如果参考块位置符合预期，则为共线延续
                                is_collinear_cont = True
                        
                        if is_collinear_cont: # 如果可以合并
                            curr_seg['q_end_merged_0based_open'] = q_e_bp # 更新查询结束位置
                            # 更新参考序列的起始和结束位置，取合并片段的最小起始和最大结束
                            curr_seg['ref_s_merged_0based'] = min(curr_seg['ref_s_merged_0based'], r_s_bp)
                            curr_seg['ref_e_merged_0based_open'] = max(curr_seg['ref_e_merged_0based_open'], r_e_bp)
                            curr_seg['last_q_idx'] = q_idx_seg
                            curr_seg['last_r_idx'] = r_pos_seg
                        else: # 如果不能合并，则完成前一个片段，开始新的片段
                            merged_segments_output.append(
                                (curr_seg['q_start_merged_0based'], 
                                 curr_seg['q_end_merged_0based_open'],
                                 curr_seg['ref_s_merged_0based'], 
                                 curr_seg['ref_e_merged_0based_open']))
                            curr_seg = { # 开始新的合并片段
                                'q_start_merged_0based': q_s_bp,
                                'q_end_merged_0based_open': q_e_bp,
                                'ref_s_merged_0based': r_s_bp,
                                'ref_e_merged_0based_open': r_e_bp,
                                'type': node_type,
                                'last_q_idx': q_idx_seg,
                                'last_r_idx': r_pos_seg
                            }
                
                if curr_seg: # 添加最后一个处理的片段
                    merged_segments_output.append(
                        (curr_seg['q_start_merged_0based'], 
                         curr_seg['q_end_merged_0based_open'],
                         curr_seg['ref_s_merged_0based'], 
                         curr_seg['ref_e_merged_0based_open']))

            if not merged_segments_output:
                print("[]") # 如果没有合并的片段，打印空列表
            else:
                # 格式化输出合并后的片段坐标字符串
                #print(merged_segments_output)
                
                end_idx = len(merged_segments_output) - 1
                begin_q = merged_segments_output[end_idx][0]
                end_q   = merged_segments_output[end_idx][1]
                begin_r = merged_segments_output[end_idx][2]
                end_r   = merged_segments_output[end_idx][3]

                dl = min(query_s_len - end_q, ref_s_len - end_r)
                merged_segments_output = [list(segment) for segment in merged_segments_output]
                merged_segments_output[end_idx][1] = end_q + dl
                merged_segments_output[end_idx][3] = end_r + dl
                #print(merged_segments_output)
                fmt_tuples = [f" ( {s[0]}, {s[1]}, {s[2]}, {s[3]} )" for s in merged_segments_output]
                output_str = "["
                for i, t_str in enumerate(fmt_tuples):
                    if i == 0:
                        output_str += t_str
                    else:
                        output_str += " ,  " 
                        output_str += t_str
                output_str += " ]"
                print(output_str)
                # 使用 calculate_value 计算最终分数
                calculated_value_score = calculate_value(output_str, ref_s, query_s)
                print(f"最长路径的计算分值: {calculated_value_score}")
        else: # 如果没有找到最长路径
            print("未找到有效的最长对齐路径。")
            print("[]")
            calculated_value_score = calculate_value("[]", ref_s, query_s) # 对空路径计算分数应为0
            print(f"计算分值 (无路径): {calculated_value_score}")

    # --- 为SVG可视化准备数据 ---
    # 将原始匹配中的 ref_chunk_start (碱基对位置) 转换为 ref_chunk_idx (块索引)
    fwd_matches_grid = []
    if fwd_matches_raw:
        fwd_matches_grid = [(q_idx, r_start // chunk_sz, score) 
                                        for q_idx, r_start, score in fwd_matches_raw]

    rc_matches_grid = []
    if rc_matches_raw:
        rc_matches_grid = [(q_idx, r_start // chunk_sz, score) 
                                   for q_idx, r_start, score in rc_matches_raw]

    # 处理转换后的匹配以识别线段和孤立点 (用于SVG网格)
    fwd_segs_grid, fwd_pts_grid = process_matches_for_visualization(fwd_matches_grid, "forward")
    rc_segs_grid, rc_pts_grid = process_matches_for_visualization(rc_matches_grid, "rc")

    print(f"处理后用于分段显示: {len(fwd_segs_grid)} 个正向线段, {len(fwd_pts_grid)} 个正向孤立点")
    print(f"处理后用于分段显示: {len(rc_segs_grid)} 个反向互补线段, {len(rc_pts_grid)} 个反向互补孤立点")

    # 为最长路径高亮转换坐标 (将参考序列的碱基对位置转换为块索引)
    long_path_grid = []
    if longest_path_nodes: # longest_path_nodes 中的 'r' 是 ref_chunk_start (碱基对位置)
        long_path_grid = [
            {
                'q': node['q'], 
                'r': node['r'] // chunk_sz, # 转换为 ref_chunk_idx
                'type': node['type'],
                'original_score': node['original_score'],
                'path_score_at_node': node.get('path_score_at_node', 'N/A')
            } for node in longest_path_nodes
        ]

    generate_svg_grid(query_s_len, ref_s_len, 
                      fwd_segs_grid, fwd_pts_grid,
                      rc_segs_grid, rc_pts_grid,
                      chunk_sz,
                      num_q_chunks, ref_chunks_svg,
                      svg_out_f,
                      long_path=long_path_grid) # 传递转换后的最长路径数据

if __name__ == '__main__':

    # 定义实际使用的文件路径
    use_query_f = "query.txt"
    use_ref_f = "ref.txt"
    svg_path = "matches_grid_visualization.svg"

    find_and_visualize_matches(
        query_f=use_query_f,
        ref_f=use_ref_f,
        svg_out_f=svg_path,
        k=6,                # K-mer 大小
        chunk_sz=25,        # 比较的序列块大小
        kmer_thresh=22,     # K-mer 匹配阈值 (基于编辑距离的分数)
        switching_penalty=10, # 切换匹配类型的惩罚
        collinear_jump_penalty=9, # 共线跳跃的惩罚
        allowed_chunk_gap=18    # 允许合并的共线段之间的最大查询块间隔
    )



