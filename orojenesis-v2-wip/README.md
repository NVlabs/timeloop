## WORK IN PROGRESS
### Version 2 of orojenesis

#### One time setup
```
python3 -m venv .venv

source .venv/bin/activate

make install_requirements

cd .venv/bin
# Make symlink to timeloop-mapper
 
```

#### Everytime to run
```
source ./.venv/bin/activate

make jupyter

# Run the notebooks!

```
### FFMR
Inter-Layer Fusion Strategies
Orojenesis presents the Fusion-Friendly Mapping Template (FFMT) to fuse sequential matrix-matrix multiplications.
To explore more complex strategies, I've found that the FFMT can be derived from the more-general Fusion-Friendly Mapping Rule (FFMR)
The Fusion-Friendly Mapping Rule (FFMR): Place all fused loops above all unfused loops in the loop nest

* If an unfused loop is above a fused loop, but that unfused loop does not index into any fused tensors, it is OK that 
Consequence 1 of (A): Intra-layer reuse is prioritized over inter-layer reuse.
Why it is helpful: Intra-layer reuse is more effectual than inter-layer reuse: A layer may use a value many times, but a layer only propagates a value to the next layer once.
Why it follows from the FFMR: Lower-level loops see more accesses and therefore can exploit more reuse, 
Consequence 2 of (A): No partial sums may be propagated between layers.
Why it is helpful: Propagating >1 partial sums would result in more data movement and computation than summing them and propagating/computing with the total.
Why it follows from the FFMR: Any to-be-reduced ranks do not show up in the output tensor, so they can not be fused. Since they are unfused, they must go below all fused loops, and therefore sums must be completed before being propagated to the next layer.
 FFMR Capacity requirements for chained Einsums E1→EN, where Ei has a working set Si,j for tensor Ti,j (working set determined by loop bounds) and there are co-iterated ranks between all Einsums:
Additive Einsum working set Ai = Σi,j(Si,T | No co-iterated rank indexes into Ti,j)
Maximal Einsum working set Mi = Σi,j(Si,T | Any co-iterated rank indexes into Ti,j)
Total capacity requirement of E1→EN = Σi(Ai) + Maxi(Mi)
Sizing the interesting design space for chained Einsums E1→EN. To simplify calculation, assume that they are a chain of MVMs, and 1-2 ranks can be fused between each pair of Einsums
Fuse or don't fuse: #Choices = (2 to 4)#Layers - 1 (For layer 1...N-1, we can choose to fuse or not fuse each shared rank). For cases with no fused ranks, we can also decide whether to keep intermediate data local (untiled fuse) or write back (don't fuse).
Permute fused ranks: #Choices = (1 to 2)#Layers - 1 (If 2 ranks can be fused, there are 2 choices at a given point. The size of this space will likely be reduced dramatically; fused ranks need to be iterated over in the same order, and the partial ordering induced by this property will reduce the space size.)
Size of fused ranks: #Choices = (100 to 400) assuming two fused ranks of power-of-two size <= 2048, power-of-two loop bound requirements.
One limitation of Timeloop is that, in a scenario where it is finding loop bounds for a rank of size 4, it will try the first prime factor (2), the second prime factor (2), and the product (4), even though the first two bounds are identical. This can be a problem for large power-of-two loop bounds; for example, for a size of 2048, it will try the factor 1024 11 times.
Given this limitation, the number of choices Timeloop explores may be closer to #Choices = (2048) (for each 2 in the factorization, it can be included or excluded in the loop bound).
The space of intra-layer mappings: #Choices = ~1e9 × (# Layers) looking at Orojenesis logs. Layers can be optimized independently, so this doesn't scale with the number of layers.
Total: #Choices ~ (2 to 4)#Layers - 1 × (1 to 2)#Layers - 1 × (100 to 400) × 1e9 × (# Layers) ~  (2 to 8)#Layers - 1× 1e10 × (# Layers)

### Single-Operation Mappings
`timeloop.py` handles single-operation evaluations. It returns a list of all
possible mappings. Explores over:
- Fused tensors: Every possible combination
- Flipped tensor core: Whether to treat operations A/B as B/A
- Uneven tensors: Every possible combination, minus any fused tensors.

The code in `timeloop.py` and the `arch.yaml` file make several assumptions.
Notably, ranks named `head` and `batch` are special. `head` is special because
there is no reuse across attention heads, so we force head to be an outer loop.
The checks for `batch` find cases where there is no reuse across `batch` across
any tensors, which may also happen in attention heads.
- If there are any fused loops and there is a rank of size >1 with `head` in the
  name, this rank must be the outermost loop.
- If the previous condition is met and all of the following are true, then the
  `batch` dimension must be the next-outermost.
  - All tensors have a `batch` dimension of size >1.
  - A shared rank shares all `batch` dimensions.
  - There are three tensors in the operation. This check is necessary because
    the previous two checks are trivially true for elementwise operations with
    a batch dimension, which would lead to all elementwise operations having a
    `batch` outer loop, even when the `batch` dimension is not indexing all
    tensors in surrounding operations.

`mappings.py` explores all the ways to piece together single-operation mappings
to map the full computation graph.

### Multi-Operation Mapping Compatibility
Given a set of two operations 1 and 2 with mappings M1 and M2, respectively. A
pair of mappings M1 and M2 are compatible if any of the following conditions are
met:
- Case A: Either operation is not tiled fused (*i.e.,* it buffers the entire
  intermediate tensor in L2.):
  - M1 does not fuse any ranks overlapping with operation 1
- Case B: The operations are tiled fused:
  - M2 does not fuse any ranks overlapping with operation 2
  - Overlapping ranks obey the same order between operations 1 and 2. There may
    be slip in the ordering if one rank in M1 overlaps multiple ranks in M2, or
    vice versa.
  - Rank sizes need not match (unevenness is allowed within a rank), but there
    may not be any ranks that are tiled in one operation but not the other[^1].

In case A, operations are considered *untiled fused* and the overall L2
footprint is a maximum of the footprints of the two operations. In case B,
operations are considered *tiled fused* and the overall L2 footprint is the sum
of the footprints of the two operations. Additivity in footprints is transitive,
so if operation 1 is tiled fused with operation 2 and operation 2 is tiled fused
with operation 4, then the total footprint will be the sum of the footprints for
all three operations.

[^1]: For example, if operation 1 and operation 2 share tensor A and tensor A
      has ranks X, Y, Z, we may have operations 1 and 2 both have fused loops over X,Y
      of arbitrary size for both operation, or we may have loops of X, but we can't
      have a fused loop of only X for one operation and only Y for the other
      operation. We also can't have fused loops over both X and Y for one operation
      and X only for the other.

      Note that this check rules out some valid uneven mappings. For example, it
      would be valid if operation 1 had fused loops (X2, Y3) and operation 2 had
      only fused loop (X2). In this case, the fusion should be valid because
      operation 2 can wait & buffer data until Y3 iterations are finished in
      operation before it proceeds. We DO NOT admit this case because it would
      require that we also check that operation 1 generates larger tile sizes in
      the X rank than operation 2. If we have to consider tile sizes, then the
      space of fused mappings increases dramatically because we have to
      differentiate between every possible loop bound when considering whether
      two mappings are compatible. When considering loop bounds, the fused
      mapping space becomes intractible to explore (unless we come up with more
      ways to prune the space).

      If we'd like to change the code to explore this class of fused mappings,
      we must implement the following rule: If the innermost loop of operation 1
      has either no matching rank in operation 2, or it has a bigger bound
      (smaller tile size), then operation 2 must wait & buffer while operation 1
      iterates. Therefore, operation 2 must be able to buffer all the data from
      operation 1 while it iterates. For this reason, ALL loop bounds in
      operation 2's fused loops must be present in operation 1 and have bounds
      <= those in operation 1. Furthermore, they must be a perfect factor of the
      matching bound in operation 1.

      I don't think permitting this class of fused mappings would make a
      significant difference, becaues tiling one rank of a tensor has a much
      larger impact on occupancy and accesses than does tiling a second rank.

Potential extensions:
- Respect layout: If operations 1 and 2 are fused (or not fused?) they must access DRAM in the same
  order for each operation or pay a penalty.
- Investigate memory-bound kernels
- Latency? LLM embedding includes high-latency data-dependent lookups.
- Head and batch (when batch acts like a head rank) are both forced to be outer
  loops in fusion, which forces some operations to use tiled fusion. Tiled
  fusion results in additive working sets, but there is no reuse across different
  heads, so theoretically we could run each head as a separate kernel & free working
  sets between head calculation. The `HEAD_IS_TILED_FUSION` flag controls this.


