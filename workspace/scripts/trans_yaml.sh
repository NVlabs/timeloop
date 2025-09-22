#!/bin/bash

# 检查是否提供了文件路径
if [ -z "$1" ]; then
  echo "用法: $0 <yaml文件路径>"
  exit 1
fi

input_file="$1"

# 检查文件是否存在
if [ ! -f "$input_file" ]; then
  echo "错误: 文件未找到: '$input_file'"
  exit 1
fi

# 使用 awk 进行所有转换
awk '
{
    # 任务1: 将 "type: datatype" 替换为 "type: dataspace"
    if ($0 ~ /type:[[:space:]]*datatype/) {
        sub(/type:[[:space:]]*datatype/, "type: dataspace");
    }

    # 任务2: 将 factors: C4 M4... 转换为 factors: C=4 M=4...
    if ($0 ~ /factors:/) {
        # 创建当前行的副本，用于操作 factors 字符串
        line_copy = $0;

        # 移除 "factors:" 及其前面的所有内容和后面的空格
        sub(/^.*factors:[[:space:]]*/, "", line_copy);

        # 构建新的 factors 字符串
        new_factors_str = "";
        # 按一个或多个空格分割字符串
        split(line_copy, arr, /[[:space:]]+/);

        for (i in arr) {
            # 检查是否符合 "大写字母+数字" 的模式 (例如 C4, M8)
            if (arr[i] ~ /^[A-Z][0-9]+$/) {
                # 提取变量 (第一个字符) 和值 (其余部分)
                var = substr(arr[i], 1, 1);
                val = substr(arr[i], 2);
                if (new_factors_str != "") {
                    new_factors_str = new_factors_str " ";
                }
                new_factors_str = new_factors_str var "=" val;
            } else {
                # 如果不符合预期模式，则保持不变
                if (new_factors_str != "") {
                    new_factors_str = new_factors_str " ";
                }
                new_factors_str = new_factors_str arr[i];
            }
        }
        # 将原始行中的 factors 部分替换为新字符串
        sub(/factors:[[:space:]]*(.*)/, "factors: " new_factors_str);
    }

    # 任务3: 将 "inter_PE_spatial" 替换为 "PE"
    # 注意：这里使用 sub，因为它只需要替换当前行的内容。
    # 如果一行中可能出现多次 inter_PE_spatial 且都需要替换，则使用 gsub
    gsub(/inter_PE_spatial/, "PE"); 

    # 任务4: 将 "inter_PE_column_spatial" 替换为 "PE_column"
    gsub(/inter_PE_column_spatial/, "PE_column");

    print; # 打印（可能已修改的）行
}
' "$input_file"
