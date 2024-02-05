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
* `optimization_metrics`: A prioritized list of cost metrics (starting with most-significant).
The default is `[ edp ]`, and the complete set of supported metrics is:
  * `energy`
  * `delay`
  * `edp` (energy-delay product)
  * `last_level_accesses` (accesses the the last/outermost buffer level)
* `num_threads`: _All_ search heuristics are multi-threaded. The mapper module instantiates
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
* `victory_condition`: If a thread sees this many consecutive _valid_ but _suboptimal_ mappings
(i.e., mappings that have higher cost than the best mapping seen so far), it declares victory
and self-terminates. If this is set to `0`, suboptimal mappings are not used as a criterion
for thread termination. Default is `500`.
* `search_size`: If a thread encounters this many valid mappings in total, it self-terminates. If
this is set to `0`, total number of valid mappings encountered is not used as a criterion for 
thread termination. Default is `0`.
* `sync_interval`: Time interval (measured in terms of number of mappings examined) after which
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
* `linear_pruned`: A linear search that prunes the superfluous permutations of unit-factors
for each index-factorization visited. This algorithm can be used for an more efficient
exhaustive search by setting search knobs appropriately (see below).
* `random`: Randomly samples a point in the mapspace and evaluates it. By default,
the same mapping can be revisited, unless the `filter_revisits` flag is set to `True`.
However, depending on the size of the mapspace, this may be an expensive operation
because the visited filter is maintained as an unordered set instead of a bitset.
* `random_pruned`: Similar to `random`, but like `linear_pruned`, the algorithm prunes
all superfluous permutations upon visiting a specific index factorization. Because this
pruning has a cost, it may be beneficial to lock an index factorization and visit a number
of random permutations before jumping to the next random index factorization. This number
is controlled by the knob `max_permutations_per_if_visit` (default is `16`).
* `hybrid` (DEFAULT): Selects a random index factorization, prunes the superfluous permutations for
that factorization, and linearly visits the pruned permutation subspace before selecting
the next random factorization.

## Other knobs

* `log_stats`: If `True`, emit the number of valid/invalid mappings and optimal-mapping updates seen
by each thread after each successful evaluation. Default is `False`.
* `log_suboptimal`: If `True`, emit summary statistics for each evaluated mapping. If `False`, emit
summary statistics only when an optimal mapping is updated. Default is `False`.
* `live_status`: If `True`, display an ncurses-based status screen tracking statistics for each
thread. Extremely useful and informative for interactive runs. Default is `False`.
* `diagnostics`: If `True`, run the mapper in diagnostic mode (more expensive, but collects statistics
about reasons why mappings failed). Used for debugging cases where the mapper isn't able to find
any valid mappings.

## Examples

Default values (i.e., an empty `mapper` section) usually serve as a good starting point.
Here is an example that simply sets the optimization metrics to prioritize mappings
with minimum `delay`, and if two mappings have the same delay, then prioritize mappings
with lower `energy`:
```
mapper:
  optimization_metric: [ delay, energy ]
```

Here is a setup to perform an exhaustive search across the entire mapspace (with the
same optimization metrics as the above example):
```
mapper:
  optimization_metric: [ delay, energy ]
  algorithm:           linear_pruned
  timeout:             0
  victory_condition:   0
  search_size:         0
```  
