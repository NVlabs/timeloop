Exercise 2
==========

## Overview

In this exercise, we continue to use the 2-level architecture from the previous exercise, but introduce a new dimension to the problem shape: **Output Channels** (represented by the dimension `K`). As a result, the space of interesting mappings becomes richer. We also explore **tiling** the `K` dimension.

## Steps and Observations

1. Open the `prob/conv1d+oc.prob.yaml` and `map/conv1d+oc-2level-os.map.yaml` files, compare them to the corresponding files from the previous exercise, and observe how the additional dimension requires additional directives in the mapping.

2. Open the `map/conv1d+oc-2level-os-tiled.map.yaml` and compare it to the un-tiled mapping above (to be precise, the above mapping was also tiled, but in a degenerate way). What we have done here is _factorized_ the `K` dimension into two factors and placed the factors at two different storage levels. Think about how this factorization results in breaking up each data-space -- Weights, Inputs and Outputs. Does it affect all spaces?

3. Run `timeloop-model` on each mapping (be sure to also include the problem and architecture files in the invocation) and look at the statistics. Which mapping is more efficient? Why?

To run the exercise, type the following from the exercises/timeloop directory:
```
    python3 run_example.py 02_os 
    python3 run_example.py 02_os_tiled
```

4. Experiment with the following in your free lab time:
   - Tile shapes and sizes (by re-factoring `P` and `K` between Buffer and MainMemory).
   - Loop orders (by changing the permutation).

5. Consider the following:
   - Can you think names for some of these dataflows? What "stationarity" do they exhibit?
   - How many distinct mappings are there?
   - Can you find the optimal mapping?
