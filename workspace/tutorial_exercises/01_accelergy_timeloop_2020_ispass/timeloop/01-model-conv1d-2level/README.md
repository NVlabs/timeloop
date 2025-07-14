Exercise 1
==========

## Overview

In this exercise, we continue to use the 1D convolution problem, but we upgrade the architecture to use a more realistic 2-level storage hierarchy with a large backing SRAM (called `MainMemory`) and a small register file (called `Buffer`) to exploit temporal reuse. We provide 2 example mappings -- **Weight Stationary** and **Output Stationary** -- that exploit reuse in different ways.

## Steps and Observations

1. Recall that to run the Timeloop model, we need to provide an architecture, a problem and a single mapping. Thus, to run the Weight Stationary mapping, type the following from the exercises/timeloop directory:

We have provided a separate `.yaml` file for each of these inputs arranged into different sub-directories. To run the exercise, type the following from the exercises/timeloop directory:

    ```
    python3 run_example.py 01_ws
    ```

    As before, Timeloop produces a `timeloop-model.stats.txt` file with detailed execution statistics. However, also pay attention to the `timeloop-model.map.txt` file. It contains a pretty-formatted version of the loop nest corresponding to the input mapping, annotated with tile sizes for each data-space at each storage level. This file will become extremely useful later when we run the automated mapper.

    > Sidenote: Subsequent `timeloop-model` invocations will overwrite output files (`timeloop-model.stats.txt` and `timeloop-model.map.txt`). You need to manually rename and save these output files if you want to retain them for comparison. For this exercise, you may not need to do this because the summary statistics on `stdout` are sufficient.

2. Now run the Output Stationary mapping (note that the architecture and problem input files are the same as the previous invocation):

    ```
    python3 run_example.py 01_os
    ```

3. Compare the statistics from the two runs. Do you see a difference? Why or why not? Do the access count and buffer utilization numbers match the expected numbers from the tables in the Powerpoint slides?

4. Now let's try something interesting. Edit the `prob/conv1d.prob.yaml` file and change the problem size from `P:16` to `P:1920`. Next, edit both mappings to reflect this change: a mapping _must_ cover the complete iteration space of the original problem, otherwise it is illegal (Timeloop will complain). Now try running both mappings as before.
    - Hmmmm... what do you think happened here? Can you change any of the _architecture_ specs to make the mappings work?

      If you decide to increase the sizes of any storage structures, **pay attention to their `class`**, which represents the underlying design used to implement the structure. Different implementations have different sweet-spots in terms of energy efficiency. Try changing the class (look for hints elsewhere in the same YAML file) if you find energy costs to be scaling in undesirable ways.

      > Sidenote: you will learn more about these classes in the **Accelergy** section of the tutorial.

Does this exercise tell you anything about the relative pros and cons of these two mappings?
