Exercise 3
==========

## Overview

In this exercise, we continue to use the problem shape from the previous exercise, i.e., a 1D Convolution with Output Channels. However, we add more realism and complexity to our architecture by making it a 3-level hierarchy with a DRAM level acting as the main backing store. Additionally, we also learn about a new mapping technique that allows data_spaces to optionally _bypass_ one or more storage levels.

## Steps and Observations

1. Open the `arch/3level.arch.yaml` and `map/conv1d+oc-3level.map.yaml` files. Observe how the addition of a storage level in the architecture requires adding a new mapping directive corresponding to that level. The space of mappings opens up even further: you can now factor a problem dimension across 3 levels instead of 2, and you can use a different permutation at each level.

2. In all exercises so far, our mappings have created tiles of various shapes and sizes for each data-space, and we have always assumed that each data-space must have a tile resident at each storage level. However, this strict inclusivity is not required. At times, it may be beneficial for a data-space to _bypass_ a storage level---for example, if there isn't enough reuse of that data-space at that level. This opens up that level's capacity to larger tiles for other data_spaces, which may (or may not) end up being more optimal. Timeloop's mapping format allows us to express bypassing. To see an example, open the `map/conv1d+oc-3level-bypass.map.yaml`. This is just a random bypassing strategy that we created for illustration and is not based on any insights.

3. Run the Timeloop model on both mappings. Was bypassing more or less efficient in this case?

To run the exercise, type the following from the exercises/timeloop directory:
```
    python3 run_example.py 03_bypass 
    python3 run_example.py 03_no_bypass
```

4. Can you come up with a better bypassing strategy? What happens if you also change the other variables in the mapping, i.e., tiling factors and permutations? How do they interplay with bypassing options?

    If you find a better bypassing strategy than any of the provided mappings, hold on to it. In a later exercise, we will run Timeloop's automated _mapper_ on this very same problem and architecture. You can compare what you found vs. what the mapper found!
