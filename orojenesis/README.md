# Mind the Gap: Attainable Data Movement and Operational Intensity Bounds for Tensor Algorithms

This work presents *Orojenesis*  an approach to compute data movement bounds for tensor algorithms.
*Orojenesis* comprehends reuse and the ability of a buffer to exploit reuse to
reduce data movement and provides a bound that no dataflow
or mapping can possibly exceed under varying on-chip buffer
capacity constraints, including mappings that fuse a sequence
of tensor operations to exploit producer-consumer reuse.

*Orojenesis*
produces a plot that shows the relationship between a buffer’s
size and the lower data movement limit to/from the next level in
a memory hierarchy.

Please refer the [page](https://timeloop.csail.mit.edu/orojenesis) for instructions to install and run the project.
