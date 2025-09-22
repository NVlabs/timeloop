import os
from typing import Any, List
import inspect
from accelergy.utils import yaml


def plugin_notebook2script(
    python_object: Any, filename: str, return_instead_of_write: bool = False
) -> None:
    """
    Writes a plug-in from this notebook into a script so Accelergy can import
    it. This is a helper funciton so we can make Accelergy plug-ins from within
    this notebook, but generally you'd write them as separate scripts.
    """
    if isinstance(python_object, List):
        script = "\n".join(
            [
                plugin_notebook2script(p, filename, return_instead_of_write=True)
                for p in python_object
            ]
        )
        name = ", ".join([p.__name__ for p in python_object])
    else:
        functions = inspect.getmembers(python_object, inspect.isfunction)
        functions = [
            f[1] for f in functions if not f[0].startswith("_") or f[0] == "__init__"
        ]
        script = (
            "from accelergy.plug_in_interface.estimator import "
            "Estimator, actionDynamicEnergy, Estimation\n"
            "from typing import Union\n"
            "from numbers import Number\n\n"
        )
        script += f"\nclass {python_object.__name__}(Estimator):\n"
        for attr in ["name", "percent_accuracy_0_to_100"]:
            v = getattr(python_object, attr)
            script += f"    {attr} = " + (
                f'"{v}"\n' if isinstance(v, str) else f"{v}\n"
            )
        script += "\n".join([inspect.getsource(f) for f in functions])
        name = python_object.__name__
    if return_instead_of_write:
        return script
    print(f"Writing {name}. View the script at {os.path.abspath(filename)}")
    open(filename, "w").write(script)


def yaml_section(
    file: str, keys: List[str] = (), only_include_base_keys: List[str] = ()
) -> str:
    """
    Prints a section of a YAML file. This is a helper function to make it
    easier to see the contents of the YAML files we'll be using in this
    tutorial.
    """
    content = yaml.load_yaml(file)
    for key in keys:
        content = content[key] if key != "subtree" else content[key][0]
    if only_include_base_keys:
        stack = [content]
        while stack:
            node = stack.pop()
            if isinstance(node, list):
                stack.extend(node)
            else:
                for k, v in list(node.items()):
                    if k in only_include_base_keys:
                        continue
                    elif isinstance(v, dict) or isinstance(v, list):
                        stack.append(v)
                    else:
                        node.pop(k)

    return "\t" + "\n\t".join(yaml.to_yaml_string(content).splitlines())


def get_log_lines(logfile_path: str, start_from_line_containing: str) -> str:
    """Grabs lines from an Accelergy log file."""
    lines = open(logfile_path).readlines()
    if isinstance(start_from_line_containing, str):
        start_from_line_containing = [start_from_line_containing]
    start = [
        i
        for i, l in enumerate(lines)
        if all([s in l for s in start_from_line_containing])
    ]
    loglevels = ["INFO", "DEBUG", "WARNING", "ERROR"]
    result = []
    for s in start:
        end = [
            i
            for i, l in enumerate(lines)
            if i > s and l.split()[-1].strip() in loglevels
        ]
        end = end[0] if end else len(lines)
        result.append("".join(lines[s:end]))
    return "\n".join(result)
