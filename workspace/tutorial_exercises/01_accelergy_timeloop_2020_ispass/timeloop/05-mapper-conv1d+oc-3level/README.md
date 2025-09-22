Exercise 5
==========

## Overview

In this exercise, we will use Timeloop's **mapper** to automatically construct and explore the _mapspace_ (i.e., the space of legal mappings) of a problem on an architecture. To keep the mapspace manageable for this tutorial setting, we revert back to the 1D Convolution with Output Channels problem shape and the 3-level temporal architecture from Exercise 3.

## Preparation

Similar to the model, the mapper needs:
* An architecture.
* A problem.

However, unlike the model, the mapper does not need an explicit mapping (it finds those for us). Instead, it needs:
* A set of _architecture constraints_ that reflect portions of the dataflow that are fixed in the hardware (grouped under the YAML configuration key `architecture_constraints`). These are not used in this exercise.
* An optional set of _mapspace constraints_ to restrict the mapspace (grouped under the YAML configuration key `mapspace_constraints`).
* A set of options for the mapper application itself, including user-interface glags and directives to guide the search heuristics (grouped under the YAML configuration key `mapper`).

We have provided 3 alternative sets of constraints for this exercise.

Study the three alternative constraints files in `constraints`:
   1. `conv1d+oc-3level-1mapping.constraints.yaml`. In this example, we have abused the constraints system to completely "bake" every single variable of the mapspace to reduce it down to a size of 1. In fact, this single remaining mapping is identical to the mapping we used in Exercise 3 (`map/conv1d+oc-3level.map.yaml`). The syntax is nearly identical, but the semantics are subtly different. You can express inequalities in constraints (e.g., `K<=16`), and permutations can be partially constrained.
   2. `conv1d+oc-3level-freebypass.constraints.yaml`: In this example, we lock down all factorizations and permutations in the mapping, but leave bypassing un-specified. This allows the mapper to explore all bypassing options.
      > Be careful about how Bypass directives are treated by the model vs. the mapper. If you don't specify any bypass directives in a mapping, the model assumes they are set to `keep` (i.e., do not bypass). Remember that a mapping cannot have any non-determinism. However, the mapper treats anything left un-specified in the constraints as options to explore in the mapspace. If you want to lock down a specific bypass strategy, you need to create an explicit constraint to restrict the mapper.
   3. `null.constraints.yaml`: In this example, we remove all constraints and let the mapper explore the complete mapspace.

Can you mentally calculate the size of the mapspace for each of these?

Next, look at the `mapper/exhaustive.mapper.yaml` file. The directives here instruct the mapper to exhaustively explore the complete mapspace. Observe:
* The `optimization_metric` gives the mapper a prioritized list of metrics to compare mappings. In this example, `delay` is the highest priority. If two mappings have the same delay, they will be compared based on `energy`. The set of metrics supported today include `energy`, `delay`, `edp` (energy-delay product), and `last_level_accesses`, but this list can be trivially expanded to _any_ metric that the model generates (as in `stats.txt`) with just a few lines of code.
* The mapper can launch multiple threads to speed up the search. It will divide up the mapspace among all threads. If the `num_threads` directive isn't specified, the mapper will query the underlying host for the number of available CPU contexts and use that many threads.

Next, we will run the mapper, once on each of the constraints files. It is useful to envision what is going to happen:
* The mapper will construct a mapspace based on the architecture and problem specifications.
* It will then constrain the mapspace based on the provided constraints.
* Next, it will _prune_ some redundant mappings out of the mapspace.
* It will then begin selecting mappings out of the mapspace, evaluate each mapping (using the same model we were manually invoking in previous exercises), and walk through the entire mapspace exhaustively before terminating.
* Finally, it will print the stats of the optimal mapping, as well as the mapping itself in a human-readable loop-nest form.

## Steps and Observations

Run all 3 experiments.
1. To run the exercise, type the following from the exercises/timeloop directory:
      ```
      python3 run_example.py 05_baked
      python3 run_example.py 05_bypass
      ```

      1. Observe the output log. Timeloop prints out the size of the mapspace in each dimension. The total size of the mapspace is the product of all these mapspace dimensions. Everything is 1 in this example because the mapspace is completely locked down.
      2. Compare the results with the default mapping provided in Exercise 3. The numbers should be identical.
2. Now run the mapper with `conv1d+oc-3level-freebypass.constraints.yaml`. Recall that our objective here is to let the mapper find the best bypassing strategy for a given set of factorization and permutations.
      1. Observe from the log that all mapspace sizes are 1 _except_ for the `DatatypeBypass` subspace.
      2. Compare the results with what you obtained in Exercise 3. Were you able to find this mapping?
3. Now run the mapper with `null.constraints.yaml`. This may take a few seconds. If you wish, you can comment out the `num_threads` directive in `mapper/exhaustive.mapper.yaml`. You can also hit Ctrl+C to interrupt the search. It will terminate graceully and emit the optimal mapping it found so far.
      ```
      python3 run_example.py 05_null
      ```
      1. Did it find an even better mapping?

> For each of these runs, be sure to look at the generated `timeloop-mapper.map.txt`. This is the optimal mapping found by the mapper shown in a loop-nest format.

## Takeaways
Mapspace constraints are used to limit the search space (and thereby speed up the search heuristics) based on the user's intuition on a what may constitute a good mapping. In contrast, architecture constraints are used to represent hardware capabilities (or rather, incapabilities). Many hardware accelerators aren't designed to be flexible enough to execute any pattern of data-movement loop nests that a mapper can throw at them. They are restricted (with good reason) to execute a limited set of alternative shapes of mappings, or _dataflows_. This simplifies the hardware design of networks and internal state machines. Timeloop presently uses an identical specification language to describe both styles of constraints, though that may change in future.
