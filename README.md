# Timeloop

## About

Timeloop is an infrastructure that aims to provide modeling, mapping and code-generation for dense- and sparse- tensor algebra workloads on a range of accelerator architectures. It is built from two modular components:

* A fast analytical model that can emulate a range of architecture designs and provide performance and energy projections
* A mapper that that searches for an optimal mapping in the space of mappings of a tensor-algebra problem on a given architecture

## Documentation

Timeloop documentation is hosted at https://timeloop.csail.mit.edu/. The guides there cover installation, usage and examples.
For a deeper understanding of Timeloop's internals please read our [ISPASS 2019 paper](https://parashar.org/ispass19.pdf).

Timeloop version 2.0 (a.k.a. Sparseloop) provides stochastic modeling of compressed-sparse tensor algebra. This work is described in our [MICRO 2022 paper](https://www.computer.org/csdl/proceedings-article/micro/2022/627200b377/1HMSE23T13a).

Timeloop version 3.0 (a.k.a. Ruby) adds support for imperfectly-factorized mappings (described in our [ISPASS 2022 paper](https://ieeexplore.ieee.org/document/9804679)), in addition to support for spatial skews and flattened mappings.

## Tutorial

New users are strongly encouraged to complete the Timeloop [tutorial](https://accelergy.mit.edu/tutorial.html). Serially walking through the [exercises](https://github.com/Accelergy-Project/timeloop-accelergy-exercises/) from the tutorial serves as an essential hands-on introduction to the tool.

## Dependencies

Timeloop depends on the isl and barvinok libraries. In particular, barvinok version 0.41.6 (along with the pre-packaged isl library) has been tested to
build successfully with this version of Timeloop. Instructions for installing barvinok can be found in the [this link](https://barvinok.sourceforge.io/).
