# Timeloop 追踪功能使用指南

## 概述

Timeloop的追踪功能可以生成详细的point-sets访问轨迹，显示嵌套循环分析在空间-时间中访问的每个坐标。这对于深入理解映射行为非常有用，但会显著降低仿真速度。

## 启用方式

### 方法1: 使用修改后的run_example.py

```bash
# 正常运行
python run_example.py 00

# 启用追踪模式
python run_example.py 00 --enable-trace
```

### 方法2: 手动设置环境变量

```bash
export TIMELOOP_ENABLE_TRACING=1
export TIMELOOP_DISABLE_TEMPORAL_EXTRAPOLATION=1
export TIMELOOP_DISABLE_SPATIAL_EXTRAPOLATION=1

# 然后运行任何timeloop命令
python run_example.py 00
```

### 方法3: 使用便捷脚本

```bash
# 正常模式
python run_with_trace.py 00

# 追踪模式  
python run_with_trace.py 00 --enable-trace
```

## 环境变量说明

| 变量名 | 作用 | 影响 |
|--------|------|------|
| `TIMELOOP_ENABLE_TRACING=1` | 启用追踪输出 | 生成详细的访问轨迹 |
| `TIMELOOP_DISABLE_TEMPORAL_EXTRAPOLATION=1` | 禁用时间外推 | 完整仿真所有时间步骤 |
| `TIMELOOP_DISABLE_SPATIAL_EXTRAPOLATION=1` | 禁用空间外推 | 完整仿真所有空间位置 |

## 使用场景

### ✅ 适合追踪的情况
- 分析特定映射的详细行为
- 调试映射问题
- 理解数据访问模式
- 验证优化效果

### ❌ 不适合追踪的情况  
- 运行mapper搜索大量映射
- 快速原型验证
- 大规模工作负载分析
- 性能基准测试

## 输出差异

### 正常模式输出
```
Utilization = 1.00 | pJ/Compute = 2.175 | Cycles = 48
```

### 追踪模式输出
```
🔍 启用Timeloop追踪模式:
   - TIMELOOP_ENABLE_TRACING=1
   - TIMELOOP_DISABLE_TEMPORAL_EXTRAPOLATION=1
   - TIMELOOP_DISABLE_SPATIAL_EXTRAPOLATION=1

Start Parsering Layout
No Layout specified, so using bandwidth based modeling
  t/ s/ Weights: { [0:3), } Inputs: { [0:18), } Outputs: { [0:16), } 
Utilization = 1.00 | pJ/Compute = 2.175 | Cycles = 48
[详细的point-sets访问轨迹...]
```

## 性能影响

| 模式 | 速度 | 详细程度 | 推荐用途 |
|------|------|----------|----------|
| 正常模式 | 快速 | 基本统计 | 快速分析和搜索 |
| 追踪模式 | 很慢 | 极详细 | 深入调试和理解 |

## 示例命令

```bash
# 基础练习追踪
python run_example.py 00 --enable-trace

# 权重静止映射追踪
python run_example.py 01_ws --enable-trace

# 输出静止映射追踪  
python run_example.py 01_os --enable-trace

# 清除之前的输出并运行追踪
python run_example.py --clear-outputs
python run_example.py 00 --enable-trace
```

## 注意事项

1. **仅在timeloop-model上使用**: 追踪应该仅用于特定映射分析，不要与mapper一起使用
2. **性能开销**: 追踪模式会让Timeloop表现得更像周期级仿真器而非快速分析模型
3. **输出大小**: 追踪输出可能非常大，确保有足够的磁盘空间
4. **调试用途**: 主要用于理解和调试映射，不适合生产性能分析

## 查看结果

追踪结果会保存在相应练习的output目录中，例如：
```
00-model-conv1d-1level/output/
├── timeloop-model.stats.txt    # 统计信息
├── timeloop-model.map.txt      # 映射详情
└── [追踪相关文件...]           # 详细追踪数据
```
