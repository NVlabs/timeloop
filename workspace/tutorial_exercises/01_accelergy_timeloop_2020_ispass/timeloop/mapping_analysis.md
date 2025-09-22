# Weight Stationary vs Output Stationary 映射分析

## 问题规模
```
R=3 (卷积核大小)
P=16 (输出特征图大小)  
Input Size = P + R - 1 = 18
```

## Weight Stationary (WS) 分析

### 时间步分解
```
时间步 0: Weight[0] × Input[0:16]  → Output[0:16]
时间步 1: Weight[1] × Input[1:17]  → Output[0:16] (累加)
时间步 2: Weight[2] × Input[2:18]  → Output[0:16] (累加)
```

### 存储需求
- **Buffer层存储**: 
  - Weights: 1个权重值 (在每个时间步)
  - Inputs: 16个输入值 (滑动窗口)
  - Outputs: 16个输出值 (完整存储用于累加)

- **数据复用**:
  - Weight复用: ✅ 每个权重值被复用16次
  - Input复用: ❌ 输入值使用后即丢弃
  - Output复用: ❌ 需要保持用于累加

### 优势与代价
**优势**:
- Weight访存最少 (3次 vs 48次无复用的情况)
- Buffer中Weight存储需求最小

**代价**:
- 必须存储完整的Output向量用于累加
- Input需要连续供给，无法复用

## Output Stationary (OS) 对比

### 预期行为 (OS映射)
```
空间位置 0: Weight[0:3] × Input[0:3]   → Output[0]
空间位置 1: Weight[0:3] × Input[1:4]   → Output[1]  
...
空间位置15: Weight[0:3] × Input[15:18] → Output[15]
```

### OS存储需求
- **Buffer层存储**:
  - Weights: 3个权重值 (完整卷积核)
  - Inputs: 3个输入值 (当前窗口)
  - Outputs: 1个输出值 (当前计算结果)

## 权衡分析总结

| 维度 | Weight Stationary | Output Stationary |
|------|-------------------|-------------------|
| Weight存储 | 1个值/时间步 | 3个值 (完整) |
| Input存储 | 16个值 (滑动窗口) | 3个值 (当前窗口) |
| Output存储 | 16个值 (完整) | 1个值 (当前) |
| Weight复用 | ✅ 高复用 | ✅ 中等复用 |
| Input复用 | ❌ 无复用 | ✅ 相邻窗口重叠 |
| Output复用 | ❌ 累加需求 | ✅ 立即写出 |

## 实际性能影响

从Timeloop输出:
```
Utilization = 1.00 | pJ/Compute = 7.241 | Cycles = 48
```

这表明:
- 计算单元利用率100%
- 每次计算消耗7.241 pJ
- 总计算周期48个 (3个时间步 × 16个并行计算)

## 设计选择的启示

**WS适合的场景**:
- Weight参数量大，访存代价高
- Output缓存充足
- 计算密集型应用

**OS适合的场景**:  
- Input数据流大，带宽受限
- Output需要立即处理
- 存储受限的边缘设备

这种分析展示了硬件加速器设计中经典的**存储层次与数据复用的权衡**问题。
