import re
from pprint import pprint

def parse_hyper_rects_to_set(rects_str):
    """
    辅助函数：将 '{ [..:..), ... }' 格式的字符串解析为超矩形字符串的集合。
    
    *** v4 最终修复: ***
    使用更通用的正则表达式，不再强制要求结尾处有逗号。
    旧模式: r'(\[.*?:\s*.*?,\))' (错误地要求必须有 ',)')
    新模式: r'(\[.*?)\)'       (正确地匹配从'['到')'的所有内容)
    """
    if not rects_str.strip():
        return set()
    # 新的、最终修正的正则表达式
    rects = re.findall(r'\[.*?\)', rects_str) 
    # 返回一个集合，其中每个元素都是一个类似 "[...]" 的字符串
    return set(rects)

def parse_log_file(log_path, target_space_stamp):
    # (此函数其余部分与 v3 相同，我们只修复了上面的辅助函数)
    log_pattern = re.compile(
        r"^\s*t/([0-9/]+)/\s+s/([0-9/]+)/\s+"
        r"Weights:\s*\{(.*?)\}\s+"
        r"Inputs:\s*\{(.*?)\}\s+"
        r"Outputs:.*$"
    )
    structured_data = {}
    with open(log_path, 'r') as f:
        for line in f:
            match = log_pattern.match(line)
            if match:
                t_str = match.group(1).strip('/')
                s_str = match.group(2).strip('/')
                time_stamp = tuple(map(int, t_str.split('/')))
                space_stamp = tuple(map(int, s_str.split('/')))

                weights_content = match.group(3)
                inputs_content = match.group(4)
                
                weights_set = parse_hyper_rects_to_set(weights_content)
                inputs_set = parse_hyper_rects_to_set(inputs_content)
                point_set = weights_set.union(inputs_set) # <--- 现在这个 union 将会成功！

                if space_stamp not in structured_data:
                    structured_data[space_stamp] = []
                
                if point_set:
                    structured_data[space_stamp].append((time_stamp, point_set))
    
    for pe_data in structured_data.values():
        pe_data.sort(key=lambda x: x[0])
    return structured_data

def calculate_delta_stream(pe_time_series_data):
    # (此函数无需更改)
    delta_stream = []
    if not pe_time_series_data:
        return delta_stream
    prev_point_set = set()
    for time_stamp, current_point_set in pe_time_series_data:
        delta_set = current_point_set - prev_point_set
        delta_stream.append((time_stamp, delta_set))
        prev_point_set = current_point_set
    return delta_stream

# --- 主程序 ---
if __name__ == "__main__":
    # 请确保这里的路径是正确的
    LOG_FILE_PATH = "/home/arch/accelergy-timeloop-infrastructure/src/timeloop/workspace/tutorial_exercises/01_accelergy_timeloop_2020_ispass/timeloop/test.txt"
    # 关键路径PE的空间戳，请根据您的日志确认其元组长度
    CRITICAL_PATH_PE_SPACE_STAMP = (0, 0, 0, 0, 0, 0)

    all_data = parse_log_file(LOG_FILE_PATH, CRITICAL_PATH_PE_SPACE_STAMP)

    if CRITICAL_PATH_PE_SPACE_STAMP in all_data and all_data[CRITICAL_PATH_PE_SPACE_STAMP]:
        critical_path_data = all_data[CRITICAL_PATH_PE_SPACE_STAMP]
        print(f"\n--- 成功解析并筛选出 Space Stamp: {CRITICAL_PATH_PE_SPACE_STAMP} 的非空数据 ---")
        
        delta_t_stream = calculate_delta_stream(critical_path_data)

        print("\n--- Delta_t 数据需求流计算结果 ---")
        print(f"{'时间戳 (Timestamp)':<25} | {'新增数据需求 (Delta_t Set)'}")
        print("-" * 80)
        
        lines_printed = 0
        for time_stamp, delta in delta_t_stream:
            if delta:
                print(f"{str(time_stamp):<25} | ", end="")
                pprint(delta, width=120) # 增加了宽度以便更好地显示
                lines_printed += 1
        
        if lines_printed == 0:
            print("计算完成，但所有 Delta_t 均为空集 (这可能意味着数据在第一个时间步就已全部加载)。")
    else:
        print(f"\n*** 最终结论: 即使在修复后，也未能为 Space Stamp {CRITICAL_PATH_PE_SPACE_STAMP} 提取到任何有效的(非空)工作集数据。请检查日志文件和PE空间戳。***")