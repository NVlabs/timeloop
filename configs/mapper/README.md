# Mapper configuration

The mapper module is responsible for searching through the mapspace for a given
(architecture, problem) for an optimal solution. The mapper is configured via
the root YAML key:
```
mapper:
```

The mapper supports a number of selectable search heuristics. The search heuristic
can be selected via the `algorithm` key, as in:
```
mapper:
  algorithm: linear-pruned
```

Note that _all_ search heuristics are multi-threaded. The mapper module splits up the
IndexFactorization mapspace across the threads. Each thread independently follows the
specified heuristic, periodically exchanging data with other threads.

The current set of supported algorithms is as follows:
* `exhaustive`: A linear search through the mapspace. Although it is called
exhaustive, it may terminate prematurely unless the generic search knobs (see
below) are set to continue searching until each thread exhausts the mapspace allocated
to it. This algorithm should never be used except for pedagogical or debugging purposes
because it re-evaluates superfluous permutations of unit-factors.
* `linear-pruned`: A linear search that prunes the superfluous permutations of unit-factors
for each index-factorization visited. This algorithm can be used for an more efficient
exhaustive search by setting search knobs appropriately (see below).
* `random`: Randomly samples a point in the mapspace and evaluates it. By default,
the same mapping can be revisited, unless the `filter-revisits` flag is set to `True`.
However, depending on the size of the mapspace, this may be an expensive operation
because the visited filter is maintained as an unordered set instead of a bitset.
* `random-pruned`: Similar to `random`, but like `linear-pruned`, the algorithm prunes
all superfluous permutations upon visiting a specific index factorization. Because this
pruning has a cost, it may be beneficial to lock an index factorization and visit a number
of random permutations before jumping to the next random index factorization. This number
is controlled by the knob `max-permutations-per-if-visit` (default is `16`).
* `hybrid` (DEFAULT): Selects a random index factorization, prunes the superfluous permutations for
that factorization, and linearly visits the pruned permutation subspace before selecting
the next random factorization.
