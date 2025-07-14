Exercise 4
==========

## Overview

The architectures used in all prior exercises have been completely devoid of parallelism. However, as we are well aware, DNN accelerators are always designed to exploit the massive amounts of parallelism inherently present in the computation of each layer. Timeloop's architecture model allows parallelism to be expressed in a hierarchical manner. In this exercise, we introduce parallelism in one level of our example architecture's hierarchy by stamping out multiple *spatial instances* of a PE that share a single Global Buffer and backing DRAM. We also introduce another dimension to the problem shape: Input Channels.

## Steps and Observations

1. Open the `arch/3levelspatial.arch.yaml` and compare it to the previous exercise's `arch/3level.arch.yaml`. The two specifications look remarkably similar, except that this new version uses an array notation `[]` to stamp out multiple _instances_ of the RegisterFile and the MACC units.
    > Picture the (abstract) hardware topology this creates: a single Global Buffer _fans out_ to an array of parallel Register Files inside PEs. This fanout represents the amplification in parallelism between the Global Buffer and Register File levels. Just like we created **temporal** tiling levels to take advantage of amplification of _storage capacity_ between levels, we will create **spatial** tiling levels to take advantage of amplification of _parallelism_ between levels. And just like we had a large space of choices in determining factors and permutations to create temporal tiles, we similarly have a space of choices in determining factors and permutations for spatial tiles. 


2. Examine the mapping file `map/conv1d+oc+ic-3levelspatial-kp-ws.map.yaml`.
   - First, consider the file name itself. Similar to Exercise 1, the `ws` in the file name indicates that the mapping exhibits Weight Stationary behavior (in this case, at the RegisterFile level). The `kp` in the file name stands for **K-Partitioned**, i.e., we map the problem shape's `K` (i.e., Output Channel) dimension across the available parallelism on the hardware.
   - Now look at the mapping specification. Observe that there is a new mapping directive with `target:GlobalBuffer` and `type:spatial`. By convention, a spatial tiling level that describes amplification of parallelism between an outer and an inner level is associated with the _outer_ level (in this case, the Global Buffer). This new directive only has a single non-unit factor: `K16`, which means that the dataspace tiles get _partitioned_ into 16 tiles along the K-dimension.
   > Spatial partitioning often leads to potent forms of hardware reuse. In this example, partitioning along the K dimension means that the Weight and Output dataspaces get partitioned along that dimension and **distributed** to the RegisterFile, but the Input dataspace (which is unaffected by K) gets _copied_ to each RegisterFile. These Inputs can be **multicast** from the GlobalBuffer to the RegisterFiles, avoiding repeated accesses to the GlobalBuffer. Timeloop automatically detects and exploits these multicast opportunities.

3. Now examine the mapping file `map/conv1d+oc+ic-3levelspatial-cp-ws.map.yaml`.
   - This is similar to the above mapping, except that the partitioning is along the `C` (i.e., Input Channel) dimension here.
   > Consider the consequences of this partitioning strategy. Both Weights and Input Channels get split along C and distributed across the Register Files. However, Outputs are unaffected by the C dimension. This means that partial sums for the same outputs will be generated in parallel across the PEs. As they move from the RegisterFiles in the PEs to the GlobalBuffer, they can in fact be **spatially-reduced** before being placed into the GlobalBuffer. This phenomenon is the inverse of a multicast.

4. Run both mappings and examine the statistics. It may be very instructive to diff them visually. The differences are minor, but there is a notable difference in energy consumption. Can you figure out why?

To run the exercise, type the following from the exercises/timeloop directory:
```
    python3 run_example.py 04_cp 
    python3 run_example.py 04_kp
```


5. We only gave two example mappings in this exercise. Between all the factors, permutations and bypass options across the temporal and spatial tiling levels, how many choices of mappings are there? What if we increased the problem's dimension sizes?

## Takeaways

Consider the following question:
> What is the performance and energy-efficiency of a given problem on a given architecture?

We hope the exercises so far have shown that:
1. The question is only meaningful if we are also given a specific mapping.
2. For meaningfully complex problems and architectures, the space of mappings is often too large to reason about manually.

To that end, we now proceed to examples that use Timeloop's **mapper** to automatically build and explore the space of mappings.
