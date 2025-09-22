Exercise 0
==========

## Overview

This exercise runs a 1D convolution on a toy architecture with a single PE consisting of a small register and a scalar MACC unit.

## Steps

To run the exercise, we need to provide 3 inputs to the Timeloop model:
- An architecture.
- A problem.
- A mapping.

We have provided a separate `.yaml` file for each of these inputs arranged into different sub-directories. To run the exercise, type the following from the exercises/timeloop directory:

```
    python3 run_example.py 00
```

This generates the following outputs:
- Overall performance (in terms of hardware utilization fraction) and energy efficiency (in terms of pJ/MACC) are reported on `stdout`.
- Detailed statistics are saved to `timeloop-model.stats.txt`.
- A pretty-printed form of the mapping is saved to `timeloop-model.map.txt`.

## Observations

- Look at the structure of the `timeloop-model.stats.txt` file.
  - It is divided into two main sections, one for Storage and Arithmetic levels, and the other for Networks connecting those levels. The Storage and Arithmetic section is itself divided into subsections, one for each Arithmetic and Storage level in the hardware topology.
  - Each section reports the level's specs (which may be user-defined, derived or default), followed by statistics from the model's execution of the mapping.
  - Some statistics are separated out into sub-sections for each data-space (such as Weights, Inputs, etc.)
  - In these exercises, some statistics are reported as 0 because we are using a simple technology model plugin for Accelergy for this tutorial.
  - At the tail-end of the file are some aggregate statistics representing net performance and energy-efficiency. In many studies, we simply `tail` the stats file to obtain these summary statistics.
- The only storage level is `Buffer`, which both acts as the backing store for all operand and result data_spaces, as well as the sole operand source and result destination for each arithmetic operation.
- Note that tiles of "Outputs" are actually partial sums. Accesses to partial sums inside buffers are Read-Modify-Update operations. Look at the number of reads and updates of Outputs at the Buffer. Are they equal?
