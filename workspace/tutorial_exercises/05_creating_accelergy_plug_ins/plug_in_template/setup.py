from setuptools import setup

PLUG_IN_NAME = "accelergy-plug-in-template"

setup(
    name=f"{PLUG_IN_NAME}",
    version="0.4",
    description="DESCRIPTION HERE",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Electronic Design Automation (EDA)",
    ],
    keywords="accelerator hardware energy estimation CACTI",
    author="<YOUR NAME>",
    author_email="<YOUR EMAIL>",
    license="MIT",
    install_requires=["accelergy>=0.4"],
    python_requires=">=3.8",
    data_files=[
        (  # This tuple can be repeated to include multiple directories
            f"share/accelergy/estimation_plug_ins/{PLUG_IN_NAME}",
            [
                "template.py",  # Path to your Python files with the plug-ins
                "template.estimator.yaml",  # Path to the .estimator.yaml file
                # Paths to any other files you'd like to include in this directory...
            ],
        ),
    ],
    include_package_data=True,
    entry_points={},
    zip_safe=False,
)

assert False, "Please change the template for your own plug-in and remove this line."
assert (
    PLUG_IN_NAME != "my-accelergy-plug-in"
), "Please change the template for your own plug-in."
