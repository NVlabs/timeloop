import os
import inspect

THIS_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

import pytimeloop.timeloopfe.v4 as tl


def run_exercise_intmac(out_dir: str = "output"):
    start_dir = THIS_SCRIPT_DIR
    spec = tl.Specification.from_yaml_files(
        os.path.join(start_dir, "arch/arch.yaml"),
        os.path.join(start_dir, "arch/components/*.yaml"),
        os.path.join(start_dir, "prob/*.yaml"),
        os.path.join(start_dir, "mapper/*.yaml"),
        jinja_parse_data={
            "mac_class": "intmac",
            "input_weight_datawidth": 8,
            "psum_datawidth": 16,
        },
    )
    tl.call_mapper(spec, output_dir=os.path.join(start_dir, f"{out_dir}/intmac"))


def run_exercise_fpmac(out_dir: str = "output"):
    start_dir = THIS_SCRIPT_DIR
    spec = tl.Specification.from_yaml_files(
        os.path.join(start_dir, "arch/arch.yaml"),
        os.path.join(start_dir, "arch/components/*.yaml"),
        os.path.join(start_dir, "prob/*.yaml"),
        os.path.join(start_dir, "mapper/*.yaml"),
        jinja_parse_data={
            "mac_class": "fpmac",
            "input_weight_datawidth": 32,
            "psum_datawidth": 32,
        },
    )
    tl.call_mapper(spec, output_dir=os.path.join(start_dir, f"{out_dir}/fpmac"))


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
    args = parser.parse_args()

    exercise_list = [
        "intmac",
        "fpmac",
    ]

    if args.clear_outputs:
        path = os.path.abspath(THIS_SCRIPT_DIR)
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
