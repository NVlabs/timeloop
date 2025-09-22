Exercise 6
==========

## Overview

In this exercise, we present a set of detailed architecture and constraints specifications used to describe the Eyeriss accelerator architecture. The constraints include directives to model the _Row Stationary_ dataflow. We use a 7-deep loop nest representing a full Convolutional Neural Network layer as the input problem. We run the mapper on an example instance of the layer shape to find an optimal mapping on the Eyeriss architecture and derive performance, energy and other statistics from the execution.

## Steps and Observations

Take a look at the YAML files describing the problem, architecture, mapspace constraints and mapper configuration.
* The problem should be fairly straightforward to understand, but note the use of a new directive called `coefficients`, which are used to set up the _stride_ and _dilation_ parameters used in index calculation.
* The architecture is similar to that from exercise 4, with the following differences:
  * There are 256 PEs instead of 16.
  * Inside each PE is a set of 3 register files, one each for Weights, Inputs and Outputs. Bypass directives in the constraints are used to force each data-space into its designated register file.
  * There is a `DummyBuffer` level between the GlobalBuffer and PEs. This is fake buffer with 0 size that bypasses all data. It is a simulation hack used to work around a temporary spatial-mapping limitation.
* The mapspace constraints may look daunting, but they follow the same rules as the constraints in the previous exercise and set up factor, permutation and bypass constraints for each temporal and spatial tiling level in the architecture.
* The mapper configuration contains a new flag called `live_status`. This triggers an ncurses-based UI screen that allows you to track the activity of all mapper threads.

To run the exercise, type the following from the exercises/timeloop directory:
```
  python3 run_example.py 06
```

The status screen shows the following per-thread statistics:
* Total mappings visited.
* Invalid mappings rejected (because of capacity violations or spatial fanout violations).
* Valid mappings evaluated.
* Consecutive invalid mappings rejected.
* Sub-optimal valid mappings evaluated since the last update of the optimal mapping.
* Stats for the optimal mapping seen thus far.

The mapper terminates gracefully when any of the following conditions are met:
* Every mapping in the mapspace has been visited.
* Number of valid mappings evaluated >= `search_size`.
* Consecutive invalid mappings rejected >= `timeout`.
* Consecutive sub-optimal valid mappings evaluated >= `victory_condition`.
* SIGINT (Ctrl+C) is detected.

Please experiment with the configuration files to get comfortable with the tool. We will have a combined Timeloop+Accelergy lab exercise based on this architecture after the Accelergy presentation.
