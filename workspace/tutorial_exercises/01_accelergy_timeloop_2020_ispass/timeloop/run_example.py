import os
import inspect

THIS_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

import pytimeloop.timeloopfe.v4 as tl


def setup_trace_environment(enable_trace=False):
    """设置Timeloop追踪环境变量"""
    if enable_trace:
        os.environ["TIMELOOP_ENABLE_TRACING"] = "1"
        os.environ["TIMELOOP_DISABLE_TEMPORAL_EXTRAPOLATION"] = "1"
        os.environ["TIMELOOP_DISABLE_SPATIAL_EXTRAPOLATION"] = "1"
        print("🔍 启用Timeloop追踪模式:")
        print("   - TIMELOOP_ENABLE_TRACING=1")
        print("   - TIMELOOP_DISABLE_TEMPORAL_EXTRAPOLATION=1")
        print("   - TIMELOOP_DISABLE_SPATIAL_EXTRAPOLATION=1")
        print("   注意: 追踪模式会显著降低仿真速度，但提供更详细的分析")
        print()
    else:
        # 清除环境变量（如果存在）
        for var in ["TIMELOOP_ENABLE_TRACING", "TIMELOOP_DISABLE_TEMPORAL_EXTRAPOLATION", "TIMELOOP_DISABLE_SPATIAL_EXTRAPOLATION"]:
            if var in os.environ:
                del os.environ[var]


def run_exercise_00(out_dir: str = "output"):
    start_dir = os.path.join(THIS_SCRIPT_DIR, "00-model-conv1d-1level")
    spec = tl.Specification.from_yaml_files(
        os.path.join(start_dir, "arch/*.yaml"),
        os.path.join(start_dir, "map/*.yaml"),
        os.path.join(start_dir, "prob/*.yaml"),
    )
    tl.call_model(spec, output_dir=os.path.join(start_dir, f"{out_dir}"))


def run_exercise_01_ws(out_dir: str = "output"):
    start_dir = os.path.join(THIS_SCRIPT_DIR, "01-model-conv1d-2level")
    # Weight stationary
    spec = tl.Specification.from_yaml_files(
        os.path.join(start_dir, "arch/*.yaml"),
        os.path.join(start_dir, "map/conv1d-2level-ws.map.yaml"),
        os.path.join(start_dir, "prob/*.yaml"),
    )
    tl.call_model(spec, output_dir=os.path.join(start_dir, f"{out_dir}/ws"))


def run_exercise_01_os(out_dir: str = "output"):
    start_dir = os.path.join(THIS_SCRIPT_DIR, "01-model-conv1d-2level")
    # Output stationary
    spec = tl.Specification.from_yaml_files(
        os.path.join(start_dir, "arch/*.yaml"),
        os.path.join(start_dir, "map/conv1d-2level-os.map.yaml"),
        os.path.join(start_dir, "prob/*.yaml"),
    )
    tl.call_model(spec, output_dir=os.path.join(start_dir, f"{out_dir}/os"))


def run_exercise_02_os(out_dir: str = "output"):
    start_dir = os.path.join(THIS_SCRIPT_DIR, "02-model-conv1d+oc-2level")
    # Output stationary
    spec = tl.Specification.from_yaml_files(
        os.path.join(start_dir, "arch/*.yaml"),
        os.path.join(start_dir, "map/conv1d+oc-2level-os.map.yaml"),
        os.path.join(start_dir, "prob/*.yaml"),
    )
    tl.call_model(spec, output_dir=os.path.join(start_dir, f"{out_dir}/no-tiling"))


def run_exercise_02_os_tiled(out_dir: str = "output"):
    start_dir = os.path.join(THIS_SCRIPT_DIR, "02-model-conv1d+oc-2level")
    # Output stationary tiled
    spec = tl.Specification.from_yaml_files(
        os.path.join(start_dir, "arch/*.yaml"),
        os.path.join(start_dir, "map/conv1d+oc-2level-os-tiled.map.yaml"),
        os.path.join(start_dir, "prob/*.yaml"),
    )
    tl.call_model(spec, output_dir=os.path.join(start_dir, f"{out_dir}/tiling"))


def run_exercise_03_no_bypass(out_dir: str = "output"):
    start_dir = os.path.join(THIS_SCRIPT_DIR, "03-model-conv1d+oc-3level")
    # No bypass
    spec = tl.Specification.from_yaml_files(
        os.path.join(start_dir, "arch/*.yaml"),
        os.path.join(start_dir, "map/conv1d+oc-3level.map.yaml"),
        os.path.join(start_dir, "prob/*.yaml"),
    )
    tl.call_model(spec, output_dir=os.path.join(start_dir, f"{out_dir}/no-bypass"))


def run_exercise_03_bypass(out_dir: str = "output"):
    start_dir = os.path.join(THIS_SCRIPT_DIR, "03-model-conv1d+oc-3level")
    # Output stationary tiled
    spec = tl.Specification.from_yaml_files(
        os.path.join(start_dir, "arch/*.yaml"),
        os.path.join(start_dir, "map/conv1d+oc-3level-bypass.map.yaml"),
        os.path.join(start_dir, "prob/*.yaml"),
    )
    tl.call_model(spec, output_dir=os.path.join(start_dir, f"{out_dir}/bypass"))


def run_exercise_04_cp(out_dir: str = "output"):
    start_dir = os.path.join(THIS_SCRIPT_DIR, "04-model-conv1d+oc-3levelspatial")
    # CP mapping
    spec = tl.Specification.from_yaml_files(
        os.path.join(start_dir, "arch/*.yaml"),
        os.path.join(start_dir, "map/conv1d+oc+ic-3levelspatial-cp-ws.map.yaml"),
        os.path.join(start_dir, "prob/*.yaml"),
    )
    tl.call_model(spec, output_dir=os.path.join(start_dir, f"{out_dir}/cp"))


def run_exercise_04_kp(out_dir: str = "output"):
    start_dir = os.path.join(THIS_SCRIPT_DIR, "04-model-conv1d+oc-3levelspatial")
    # KP mapping
    spec = tl.Specification.from_yaml_files(
        os.path.join(start_dir, "arch/*.yaml"),
        os.path.join(start_dir, "map/conv1d+oc+ic-3levelspatial-kp-ws.map.yaml"),
        os.path.join(start_dir, "prob/*.yaml"),
    )
    tl.call_model(spec, output_dir=os.path.join(start_dir, f"{out_dir}/kp"))


def run_exercise_05_baked(out_dir: str = "output"):
    start_dir = os.path.join(THIS_SCRIPT_DIR, "05-mapper-conv1d+oc-3level")
    # 3-level mapping constraints
    spec = tl.Specification.from_yaml_files(
        os.path.join(start_dir, "arch/*.yaml"),
        os.path.join(
            start_dir, "constraints/conv1d+oc-3level-1mapping.constraints.yaml"
        ),
        os.path.join(start_dir, "prob/*.yaml"),
        os.path.join(start_dir, "mapper/*.yaml"),
    )
    tl.call_mapper(spec, output_dir=os.path.join(start_dir, f"{out_dir}/baked"))


def run_exercise_05_bypass(out_dir: str = "output"):
    start_dir = os.path.join(THIS_SCRIPT_DIR, "05-mapper-conv1d+oc-3level")
    #  Bypass mapping constraints
    spec = tl.Specification.from_yaml_files(
        os.path.join(start_dir, "arch/*.yaml"),
        os.path.join(
            start_dir, "constraints/conv1d+oc-3level-freebypass.constraints.yaml"
        ),
        os.path.join(start_dir, "prob/*.yaml"),
        os.path.join(start_dir, "mapper/*.yaml"),
    )
    tl.call_mapper(spec, output_dir=os.path.join(start_dir, f"{out_dir}/freebypass"))


def run_exercise_05_null(out_dir: str = "output"):
    start_dir = os.path.join(THIS_SCRIPT_DIR, "05-mapper-conv1d+oc-3level")
    #  Null constraints
    spec = tl.Specification.from_yaml_files(
        os.path.join(start_dir, "arch/*.yaml"),
        os.path.join(start_dir, "constraints/null.constraints.yaml"),
        os.path.join(start_dir, "prob/*.yaml"),
        os.path.join(start_dir, "mapper/*.yaml"),
    )
    tl.call_mapper(spec, output_dir=os.path.join(start_dir, f"{out_dir}/null"))


def run_exercise_06(out_dir: str = "output"):
    start_dir = os.path.join(THIS_SCRIPT_DIR, "06-mapper-convlayer-eyeriss")
    spec = tl.Specification.from_yaml_files(
        os.path.join(start_dir, "arch/*.yaml"),
        os.path.join(start_dir, "arch/components/*.yaml"),
        os.path.join(start_dir, "prob/*.yaml"),
        os.path.join(start_dir, "mapper/*.yaml"),
    )
    tl.call_mapper(spec, output_dir=os.path.join(start_dir, f"{out_dir}"))
def run_exercise_06_model(out_dir: str = "output"):
    start_dir = os.path.join(THIS_SCRIPT_DIR, "06-model-convlayer-eyeriss")
    spec = tl.Specification.from_yaml_files(
        os.path.join(start_dir, "arch/*.yaml"),
        os.path.join(start_dir, "arch/components/*.yaml"),
        os.path.join(start_dir, "prob/*.yaml"),
        os.path.join(start_dir, "map/*.yaml"),
    )
    tl.call_model(spec, output_dir=os.path.join(start_dir, f"{out_dir}"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run timeloop exercises")
    parser.add_argument(
        "exercise",
        type=str,
        help="Exercise to run. 'All' to run all exercises",
        default="",
        nargs="*",
    )
    parser.add_argument(
        "--generate-ref-outputs", action="store_true", help="Generate reference output"
    )
    parser.add_argument(
        "--clear-outputs", action="store_true", help="Clear output directories"
    )
    parser.add_argument(
        "--enable-trace", action="store_true", 
        help="Enable Timeloop tracing with disabled extrapolation (slower but more detailed)"
    )
    args = parser.parse_args()

    # 设置追踪环境
    setup_trace_environment(args.enable_trace)

    exercise_list = [
        "00",
        "01_ws",
        "01_os",
        "02_os",
        "02_os_tiled",
        "03_no_bypass",
        "03_bypass",
        "04_cp",
        "04_kp",
        "05_baked",
        "05_bypass",
        "05_null",
        "06",
        "06_model"
    ]

    if args.clear_outputs:
        prefixes = list(set([e.split("_")[0] for e in exercise_list]))

        for p in prefixes:
            path = os.path.abspath(os.path.join(THIS_SCRIPT_DIR, p + "*"))
            os.system(f"cd {path} ; rm -rf output")
            os.system(f"cd {path} ; rm -rf ref-output")
        exit()

    def run(target):
        func = globals()[f"run_exercise_{target}"]
        # Print out the function contents so users can see what the function does
        print("\n\n" + "=" * 80)
        print(f"Calling exercise {target}. Code is:")
        print(inspect.getsource(func))
        if args.generate_ref_outputs:
            func("ref-output")
        else:
            func()

    for e in args.exercise:
        if e.lower() == "all":
            for e in exercise_list:
                run(e)
        else:
            run(e)
