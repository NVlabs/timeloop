# LayoutLoop

## About

LayoutLoop is a tool based on [TimeLoop](https://timeloop.csail.mit.edu/). It integrates the functionalities of more accurate layout-based memory modeling.

The key contributions of SquareLoop over previous tools are:
* realistic layout-based memory model utilizing accurate dataspace-wise evaluation
* introduction of physical ranks, allowing for independent per-dataspace layout and AuthBlock specification 
* Layout-Mapping co-search algorithm

![Overview](overview.png)


## Usage

It is recommended to clone this repository inside the docker available [here](https://github.com/Accelergy-Project/accelergy-timeloop-infrastructure/tree/master), which has all required packages for building squareloop.
Running the experiments will require installing the following python packages in a virtual environment:

`pip install torch torchlens pyyaml torchvision pandas`

To build SquareLoop use the `scons` command in the root directory of the repository.

SquareLoop interface is based on that of TimeLoop with some additions.
Use `build/timeloop-model` to evaluate a workload on a given architecture provided a mapping.
In order to enable layout-based memory modeling include a layout file in the command. Example layouts can be found in the experiment results.

`build/timeloop-mapper` with provided layout will search for the best mapping using that layout. If layout is not provided the algorithm will co-search the mapping and layout.


## Code Structure

Most changes introduced by SquareLoop are to the following files:

* `src/model/buffer.cpp` -
    This file contains the code for layout and AuthBlock modeling. The entrypoint for that evaluation is in the function `BufferLevel::ComputeBankConflictSlowdown`, which calculates a slowdown caused by memory.

* `src/layoutspaces/layoutspace.cpp` -
    This file contains the code that creates the design space of layouts and AuthBlocks used in the Authentication-Layout-Mapping co-search algorithm.

* `src/applications/mapper/mapper-thread.cpp` -
    This file extends the mapping search by exploration of the layout and AuthBlock design space created in `layoutspace.cpp`.




## Experiment setup

We use the following files in the experiments:

* Architecture
    * SIGMA (vector256)
        * `benchmarks/arch_designs/vector_256.yaml`
    * Edge-TPU (systolic)
        * `benchmarks/arch_designs/vector_256.yaml`
        * `benchmarks/arch_designs/systolic_constraint/mapspace_XY_OS.yaml`
        * `benchmarks/arch_designs/systolic_constraint_depthwise/mapspace_XY_OS.yaml`
    * Eyeriss (eyeriss)
        * `benchmarks/arch_designs/eyeriss_like/arch/eyeriss_like.yaml`
        * `benchmarks/arch_designs/eyeriss_like/arch/components/*`
        * `benchmarks/arch_designs/eyeriss_like/constraints/*`
        * `benchmarks/arch_designs/eyeriss_like/constraints_depthwise/*`
* Workloads
    * ResNet18
        * `benchmarks/layer_shapes/resnet18/*`
    * MobileNetV3
        * `benchmarks/layer_shapes/mobv3/*`
* Mapper
    * `benchmarks/mapper/mapper_squareloop.yaml`


Enjoy! XD
