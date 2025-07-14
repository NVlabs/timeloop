#!/bin/bash

# 检查是否提供了文件路径
if [ -z "$1" ]; then
  echo "Usage: $0 <path_to_yaml_file>"
  exit 1
fi

input_file="$1"

# 检查文件是否存在
if [ ! -f "$input_file" ]; then
  echo "Error: File not found at '$input_file'"
  exit 1
fi

# 使用 awk 来删除 constraints 块
awk '
BEGIN {
    in_constraints = 0; # 标志，表示当前是否在 constraints 块内部
    constraints_indent = -1; # 记录 constraints: 所在行的缩进级别
}

{
    # 获取当前行的缩进
    match($0, /^[[:space:]]*/);
    current_indent = RLENGTH; # RLENGTH 是匹配到的空白字符的长度

    # 检查是否进入 constraints 块
    if ($0 ~ /^[[:space:]]*constraints:/) {
        in_constraints = 1;
        constraints_indent = current_indent;
        next; # 跳过当前这一行 (constraints: 这一行)
    }

    # 如果在 constraints 块内部
    if (in_constraints) {
        # 如果当前行的缩进小于等于 constraints 块的缩进，
        # 则表示 constraints 块结束了，并且当前行是下一个有效行
        if (current_indent <= constraints_indent && current_indent != 0) { # 检查 current_indent != 0 排除顶层块
             in_constraints = 0; # 退出 constraints 模式
             constraints_indent = -1;
             print; # 打印当前行
        } else if (current_indent == 0 && $0 !~ /^[[:space:]]*$/ && in_constraints){ # 处理顶层
             in_constraints = 0; # 退出 constraints 模式
             constraints_indent = -1;
             print; # 打印当前行
        } else if ($0 ~ /^[[:space:]]*$/ && current_indent <= constraints_indent) { # 处理空行，如果空行缩进小于等于constraints缩进，也结束。
            in_constraints = 0;
            constraints_indent = -1;
            print; # 打印空行，因为它标志着块的结束
        }
        # 如果仍然在 constraints 块内部（缩进更大），则不打印当前行，跳过
        else {
            next;
        }
    } else {
        # 如果不在 constraints 块内部，则正常打印行
        print;
    }
}
' "$input_file"
