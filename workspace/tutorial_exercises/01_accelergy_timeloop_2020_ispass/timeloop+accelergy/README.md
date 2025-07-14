Timeloop + Accelergy Free Lab
==========

## Overview
In this final exercise, we show how to use features from both Timeloop and Accelergy to evaluate DNN accelerators systematically, efficiently, and accurately. This exercise combines the last exercises from the two previous sessions:

1. How to map a CNN layer onto a Eyeriss-like architecture (timeloop/06).
2. How to model a complex accelerator with user-defined components (accelergy/04).

To combine them, we will provide detailed design descriptions for the system and compound components, like in Accelergy, and descriptions of problem, mapping, and constraints, to Timeloop.

## Evaluate a complex DNN accelerator running a CNN layer
The following command uses Timeloop to find a good mapping to run on an Eyeriss-like architecture:

```
timeloop-mapper arch/eyeriss_like-int16.yaml \
                arch/components/*.yaml \
                prob/prob.yaml \
                mapper/mapper.yaml \
                constraints/*.yaml
```

Take a look at `timeloop-mapper.stats.txt` output file. What's the energy breakdown and system utilization?

## Example experiments
Using those provided files, there are many interesting experiments we can explore.

1. An Eyeriss-like architecture for floating-point data:

    Data type and precision are critical to DNN accelerators. What if we want to build an accelerator operate on different data types? The following command evaluate running an Eyeriss-like architecture with floating-point input, weights, and output:

    ```
    python3 run_example.py fpmac
    ```

    As before, Timeloop produces a `timeloop-mapper.stats.txt` file with detailed execution statistics. You should see MAC energy increases significantly, showing the cost of operating on floating-point data. Compare two arch.yaml and see how they are different from each other. What if we use the same bandwidth for the floating-point design at the PE level?

    > Note: Search for `Bandwidth throttling` to see how memory hierarchy bandwidth becomes a bottle neck when using the same bandwidth.

    Now, you can imagine doing a small research project with this framework: Implemente a MAC unit for a new data type (e.g., 8-bit FP), provide the energy consumption in Accelergy, and then use timeloop to evaluate how this new data type improves overall performance and energy efficiency.

    ```
    python3 run_example.py intmac
    ```

2. Using different DRAM type:

    Main memory is often the bottleneck of performance and energy efficiency. Architects might want to improve performance and energy efficiency of their accelerators by using an advanced DRAM technology that has lower energy consumption. You can do that easily here by modifying the type in DRAM to other types (e.g., "HBM2") and rerun timeloop. Observe how the DRAM energy changes.

    You can also imagine that if Accelergy can use plug-ins for other memory technologies, we can quickly evaluate how Eyeriss performs with new technologies.

There are many more interesting research questions regarding DNN accelerators. Timeloop+Accelergy is a great framework to help researchers answer those questions. 
