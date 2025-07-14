Exercise 4
==========

## Overview

This exercise runs a energy estimation on an eyeriss-like architecture
that is described with user-defined compound components.
## Steps

To run the exercise, we need to provide 3 inputs to the Accelergy:
- An architecture description.
- A compound component description.
- An action counts.


We have provided the `.yaml` files in the input folder. To run this exercise, type: 

```
    accelergy input/ -o output/ 
```

This generates the following output in folder `output/`
- The received input files and the important evaluation progress are reported on `stdout`.
- The energy reference tables (ERTs) (pJ/action) for the components in the architecture `ERT.yaml`.
- The energy reference table summary for the components in the architecture `ERT_summary.yaml`.
- The area reference tables (ERTs) (umm^2/component) for the components in the architecture `ART.yaml`(note that the DRAM area is a place holder as it is not on-chip).
- The area reference table summary for the components in the architecture `ART_summary.yaml`.
- The flattened architecture that describes the class and attributes of the components in
the architecture is saved to `flattened_architectue.yaml`.
- The estimation (pJ) for the primitive components in the design are saved in `energy_stimation.yaml`.


## Observations

- Note that the `input/architecture.yaml` is much more complicated than the previous examples. The architecture contains three main parts, 2 global buffers, 4 NoCs for different data types, and the 168 PEs, each contain 3 local buffers and a MAC unit.
- The compound components can be specified separately to avoid long and tedious compound component descriptions. Examine the files, what is the relationship between different compound components? Are they dependent or independent? 
- What will the architecture description look like if we use primitive components to describe the architecture?
- Examine the `input/action_counts.yaml`, which action is specific to zero-gating? Can you tell which PE processes the most sparse data? 
- What will the action_counts look like if we use record the runtime activity of each ptimitive component in the design?
- Examine the `output/estimation.yaml`, do you see the PE processing the most sparse data have the least energy consumption? Can you tell which components are most relevent to the zero-gating optimization?


## Takeaways
- We can define our own compound components and use them to define the architecture.
- Modeling the design using compound components 
  - Reduces the architecture description and action counts complexity 
  - Makes architecture redesigns much faster as no new action counts need to be 
    generated. 
- Design-specific optimizations can be very well reflected using the fine-grained action types.
