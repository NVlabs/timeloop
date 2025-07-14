Sparse Tensor Core Like Architecture
----------------------------
This folder contains an architecture based on the publicly available information on the sparse tensor core design that supports 2:4 structured sparsity.


Q&As
----------------------------
#### 1. Is this design the exact sparse tensor core design in NVIDIA GPU?

  **No, this design is just an acclerator emulation of the sparse tensor core based on the publicly available information.** Many parameters are adjusted, e.g., the available bandwidth, to model a standalone accelerator design. 

#### 2. Does the provide constraints work for conv kernels?

  No, the sparse tensor core design treats all problems as matrix multiplications, so the constraints are setup based on the **matrix multiplication** problem shape. **Please use example layers under `../layer_shapes/MM` to run the design**. `A` tensor is our 2:4 structured sparse weight tensor in all of the problems specs. We also provide an example problem in this directory: `prob/prob.yaml`.

#### 3. What does `skipping_spatial` in `sparse-opt/sparse-opt.yaml` mean?

  `skipping_spatial` refers to skipping on spatial instances, i.e., if a specific hardware instance is assigned to tiles with all zeros, use it for other tiles.
By doing that, we allow more processing parallelism.

#### 4. Where does the 2:4 structure along the channels come into the picture?

  Please note that our mapspace search contraints mandates the that innermost loop is on the channel dimension, i.e., `K`.
As a result, the 50% fixed structured sparsity is always assigned to the channel dimension. If you would like to change the setup, do make sure that 
the innermost loop bound is a mutiple of 4 in your problem or mapspace constraint.

#### 5. More studies?

  Please refer to the case study in our [paper](https://arxiv.org/pdf/2205.05826.pdf) for a more invovled case study that looks at variations of sparse tensor core.
