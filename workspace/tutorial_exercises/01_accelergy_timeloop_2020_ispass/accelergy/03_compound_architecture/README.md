Exercise 3
==========

## Overview

This exercise runs a energy estimation on an architecture
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
the architecture is saved to `flattened_architecture.yaml`.
- The estimation (pJ) for the primitive components in the design are saved in `estimation.yaml`.


## Observations

- Note that the `input/compound_component.yaml` contains an `includedir` keyword that indcludes all the files in the `component` subfolder.
- The compound components can be specified separately to avoid long and tedious compound component descriptions. Examine the files, what are the subcomponents of the compound components? How are the compound actions defined?
- Are the components in the architecture description compound components or primitive components?
- Are the actions counts in terms of compound action or primitive components? If we change the definitions of the compound components in the compound component description files, do we need to generate a new set of action counts?
- According to the compound description, try describe the architecture with primitive components, would it be tedious? If we change add a fifo to store the output data from a local buffer, what do we need to change in the architecture description? What do we need to change in the action counts file?


## Takeaways
- We can define our own compound components and use them to define the architecture.
- Modeling the design using compound components 
  - Reduces the architecture description and action counts complexity 
  - Makes architecture redesigns much faster as no new action counts need to be 
    generated. 
