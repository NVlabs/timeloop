#!/usr/bin/env python3
"""
Timeloop Trace运行示例

这个脚本演示了如何使用Timeloop的追踪功能。
追踪模式会生成详细的point-sets访问轨迹，但会显著降低仿真速度。

使用方法:
1. 正常运行: python run_with_trace.py 00
2. 启用追踪: python run_with_trace.py 00 --enable-trace

追踪输出会包含:
- 详细的空间-时间坐标访问轨迹
- 禁用时间和空间外推的完整仿真
- 更精确但更慢的分析结果
"""

import os
import sys


def run_with_trace(exercise_id, enable_trace=False):
    """运行指定练习，可选择启用追踪"""

    # 设置追踪环境变量
    if enable_trace:
        os.environ["TIMELOOP_ENABLE_TRACING"] = "1"

        print("🔍 Timeloop追踪模式已启用")
        print("=" * 50)
        print("环境变量设置:")
        print("  TIMELOOP_ENABLE_TRACING = 1")
        print("")
        print("⚠️  注意: 追踪模式会显著降低仿真速度")
        print("   建议仅在需要详细分析特定映射时使用")
        print("=" * 50)
        print("")
    else:
        # 清除追踪环境变量
        for var in [
            "TIMELOOP_ENABLE_TRACING",
            "TIMELOOP_DISABLE_TEMPORAL_EXTRAPOLATION",
            "TIMELOOP_DISABLE_SPATIAL_EXTRAPOLATION",
        ]:
            if var in os.environ:
                del os.environ[var]
        print("🏃 正常模式运行")
        print("")

    # 调用原始脚本
    cmd = f"python run_example.py {exercise_id}"
    if enable_trace:
        cmd += " --enable-trace"

    print(f"执行命令: {cmd}")
    print("=" * 50)

    # 执行命令
    exit_code = os.system(cmd)

    if exit_code == 0:
        print("")
        print("=" * 50)
        print("✅ 执行完成!")

        if enable_trace:
            print("")
            print("📊 追踪输出文件应该包含:")
            print("  - 更详细的仿真日志")
            print("  - point-sets访问轨迹")
            print("  - 空间-时间坐标映射详情")

        # 显示输出目录
        exercise_dirs = {
            "00": "00-model-conv1d-1level",
            "01_ws": "01-model-conv1d-2level",
            "01_os": "01-model-conv1d-2level",
        }

        if exercise_id in exercise_dirs:
            output_dir = f"{exercise_dirs[exercise_id]}/output"
            print(f"  📁 输出目录: {output_dir}")

    else:
        print(f"❌ 执行失败 (退出代码: {exit_code})")


def main():
    if len(sys.argv) < 2:
        print("用法: python run_with_trace.py <exercise_id> [--enable-trace]")
        print("")
        print("练习选项:")
        print("  00        - 基础1D卷积1层模型")
        print("  01_ws     - 权重静止2层模型")
        print("  01_os     - 输出静止2层模型")
        print("  all       - 运行所有练习")
        print("")
        print("示例:")
        print("  python run_with_trace.py 00")
        print("  python run_with_trace.py 00 --enable-trace")
        sys.exit(1)

    exercise_id = sys.argv[1]
    enable_trace = "--enable-trace" in sys.argv

    run_with_trace(exercise_id, enable_trace)


if __name__ == "__main__":
    main()
