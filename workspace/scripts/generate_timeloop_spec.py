if __name__ == "__main__":
    import pytimeloop.timeloopfe.v4 as tl
    import os

    THIS_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    TOP_PATH = f"{THIS_SCRIPT_DIR}/../tutorial_exercises/02_interface_and_design_space_exploration_2024/top.yaml.jinja"
    TARGET_FILE = f"{THIS_SCRIPT_DIR}/../timeloop_spec.yaml"
    spec = tl.Specification.from_yaml_files(TOP_PATH)
    with open(TARGET_FILE, "w") as f:
        f.write(tl.doc.get_property_yaml(spec))
    print(f"Timeloop specification written to {TARGET_FILE}")
