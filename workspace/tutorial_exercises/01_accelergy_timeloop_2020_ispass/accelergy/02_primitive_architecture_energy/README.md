Exercise 2
==========

## Overview

This exercise runs a energy estimation on a simple architecture using the provided ERT and an action counts.

## Steps

To run the exercise, we need to provide 3 inputs to the Accelergy:
- An ERT.
- Action counts.
- A flattened architecture/ the original hierarchical architecture.


We have provided the `.yaml` files in the input folder. To run this exercise, type: 

```
    accelergy input/ -o output/ -f energy_estimation
```

This generates the following output in folder `output/`
- The received input files and the important evaluation progress are reported on `stdout`.
- The estimation (pJ) for the primitive components in the design are saved in `energy_estimation.yaml`.


## Observations

- The `input/ERT.yaml` file contains a top-level key `ERT` that is important 
  for Accelergy to identify the file content.
- The `input/flattened_architecture.yaml` file contains a top-level key `flattened_architecture` that is important 
  for Accelergy to identify the file content. Try to change the component names in the action counts and rerun,
  does Accelergy complain? Why is this check can sometimes be useful?
  This file is optional and is used to check if the component names in the action counts are legal.
- The `output/estimation.yaml` file contains the energy estimations of all components specified in the action counts.
- Is this step faster than the energy reference table (ERT) generation step? Will this step be even faster if the design 
is large and complicated? 
- Try play with the counts of each action in the `input/action_counts.yaml`, do we need to regenerate the ERT after the 
count is updated?  Why is it useful to have energy calculator independent from ERT generator?

