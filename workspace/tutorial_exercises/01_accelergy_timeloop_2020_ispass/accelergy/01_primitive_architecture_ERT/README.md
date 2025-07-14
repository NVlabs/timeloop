Exercise 1
==========

## Overview

This exercise runs a energy reference table (ERT) generation on a simple architecture 
that is described by primitive components.

## Steps

To run the exercise, we need to provide 1 input to the Accelergy:
- An architecture.


We have provided the `.yaml` file in the input folder. To run this exercise, type: 

```
    accelergy input/ -o output/ 
```

This generates the following output in folder `output/`
- The received input files and the important generation progress are reported on `stdout`.
- the ERTs (pJ/action) for the primitive components in the design are saved in `ERT.yaml`.
- A summary of the ERTs are saved in `ERT_summary.yaml`.
- The area reference tables (ERTs) (umm^2/component) for the components in the architecture `ART.yaml`(note that the DRAM area is a place holder as it is not on-chip).
- The area reference table summary for the components in the architecture `ART_summary.yaml`.
- A summary of the parsed architecture is saved in `flattened_arch.yaml`.

## Observations

- The `input/architecture.yaml` file contains a top-level key `architecture` that is important 
  for Accelergy to identify the file content.
- The `output/flattned_architecture.yaml` file contains interpreted information for each component in the design. 
  Note that all the default attributes that are not specified in the architecture description is added. This file can
  be used to check if Accelergy correctly understands the architecture description.
- The `output/ERT.yaml` file contains the detailed ERTs for all the components in the design. For each action, if the 
  action has runtime arguments, the energy is reported for each (set of) possible argument value(s). How does the arguments
  affect energy/action values? Is it important to have fine-grained action types?
- The `output/ERT_summary.yaml` file contains the a summarized ERT of the components in the design. For each action of 
  a component, if the action has run time arguments, it summarizes the min, max, and average of the action energy;
  otherwise, it reports the energy of the action.