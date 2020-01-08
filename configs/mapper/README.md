# Mapper configuration

The mapper module is responsible for searching through the mapspace for a given
(architecture, problem) for an optimal solution. The mapper is configured via
the root YAML key:
```
mapper:
```

## Main knobs

The main configuration knobs for the mapper are:
* `algorithm`: Selects one of several search heuristics (see below). Default is `hybrid`.
* `optimization-metrics`: A prioritized list of cost metrics (starting with most-significant).
The list of supported metrics is:
  * `energy`
  * `delay`
  * `edp`
  * `last-level-accesses`
* `num-threads`: _All_ search heuristics are multi-threaded. The mapper module instantiates
the given number of threads and divvies up the IndexFactorization mapspace across them. Each
thread independently follows the specified heuristic, periodically exchanging data with other
threads. If left unspecified, the mapper queries the underlying host platform for the
available hardware concurrency and instantiates that many threads.

## Tuning search termination conditions

The following knobs are used to tune the search heuristics. Specifically, they determine
when a search thread either declares victory or gives up and terminates.

* `timeout`: If a thread sees this many consecutive invalid mappings, it gives up and
self-terminates. If this is set to `0`, invalid mappings are ignored and not used as a criterion
for thread termination. Default is `1000`.
* `victory-condition`: If a thread sees this many consecutive _valid_ but _suboptimal_ mappings
(i.e., mappings that have higher cost than the best mapping seen so far), it declares victory
and self-terminates. If this is set to `0`, suboptimal mappings are not used as a criterion
for thread termination. Default is `500`.
* `search-size`: If a thread encounters this many valid mappings in total, it self-terminates. If
this is set to `0`, total number of valid mappings encountered is not used as a criterion for 
thread termination. Default is `0`.
* `sync-interval`: Time interval (measured in terms of number of mappings examined) after which
each thread shares the best mapping it has seen so far (and its cost) with other threads. This
is not a full barrier - each thread simply syncs with a globally-shared best mapping. Default is
`0` (threads operate independently and do not sync, except at the end after all threads have
terminated).

## Search algorithms

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
