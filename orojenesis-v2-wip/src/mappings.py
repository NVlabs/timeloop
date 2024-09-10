# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import copy
import functools
from collections import defaultdict
import itertools
import pickle as pkl
from typing import Callable, Iterable, List, Set, Tuple, Union

from tensors import Tensor
from paretos import Pareto
from operations import SharedRank, Operation, Rank, OperationList
import paretos

from util import *

HEAD_IS_TILED_FUSION = False


class FusedMapping:
    def __init__(
        self,
        fused_tensors: List[Tensor],
        fused_loops: List[Tuple[SharedRank, Rank]],
        op: Union[Operation, List[Operation]],
        fusion_string: str,
        mapping_result: Pareto,
    ):
        # IMMUTABLES
        self._fused_tensors: List[Tensor] = list(fused_tensors)
        self._op: Set[Operation] = [op] if not isinstance(op, list) else op
        self._tensor_names: Set[str] = {t.name for o in self.op for t in o.tensors}
        self._fused_ranks: List[Rank] = [r for _, r in fused_loops]
        self._rank_names: Set[Rank] = {r.name for o in self.op for r in o.ranks}
        self._fused_loops = [
            f
            for f in fused_loops
            if HEAD_IS_TILED_FUSION or not any("head" in r.name for r in f[0].ranks)
        ]

        self._co_tiled_tensors = set(
            [
                t.name
                for shared_rank, _ in self._fused_loops
                for t in self._fused_tensors
                if t in shared_rank.tensors
            ]
        )
        self._fused_tensor_names = set(t.name for t in self.fused_tensors)
        self._fusion_string = fusion_string
        self.mapping_result = mapping_result
        self._as_str = self._get_str()
        self._hashed = hash(self._as_str)

    @property
    def fused_tensors(self):
        return self._fused_tensors

    @property
    def tensor_names(self):
        return self._tensor_names

    @property
    def fused_ranks(self):
        return self._fused_ranks

    @property
    def rank_names(self):
        return self._rank_names

    @property
    def op(self):
        return self._op

    @property
    def fusion_string(self):
        return self._fusion_string

    @property
    def fused_loops(self):
        return self._fused_loops

    @property
    def df(self):
        return self.mapping_result.df

    @df.setter
    def df(self, value):
        self.mapping_result.df = value

    @property
    def fused_tensor_names(self):
        return [t.name for t in self.fused_tensors]

    def is_tiled_fused_with(self, other: "FusedMapping") -> bool:
        return bool(self._co_tiled_tensors & other._co_tiled_tensors)

    def get_relevant_components(
        self,
        other: Union[Operation, List[Operation]],
        ignore_fused_ranks: bool = False,
    ):
        if not isinstance(other, list):
            other = [other]

        important_tensors = [
            t
            for t in self.fused_tensors
            if any(t.name in o.tensor_names for o in other)
        ]
        if ignore_fused_ranks or not important_tensors:
            return important_tensors, []

        first_fused_index = 0
        for l, _ in self.fused_loops:
            if any(l.overlaps(l2) for o in other for l2 in o.shared_ranks):
                break
            first_fused_index += 1
        fused_loops = self.fused_loops[first_fused_index:]

        return important_tensors, fused_loops

    @functools.lru_cache(10000)
    def strby(
        self,
        other: Union[Operation, Tuple[Operation]],
        ignore_fused_ranks: bool = False,
        ignore_fused_rank_size: bool = True,
    ):
        # BUGFIX: MAKE SURE THAT SIZE IS INCLUDED FOR ALL BUT THE
        # LAST (INNERMOST) FUSED LOOP
        if isinstance(other, tuple):
            return " ".join(
                self.strby(
                    o,
                    ignore_fused_ranks=ignore_fused_ranks,
                    ignore_fused_rank_size=ignore_fused_rank_size,
                )
                for o in other
            )

        important_tensors, fused_loops = self.get_relevant_components(
            other, ignore_fused_ranks
        )
        return " ".join(
            # Tensors unordered -> sort them
            sorted([t.name for t in important_tensors])
            # Fused loops ordered -> DO NOT SORT
            + [
                f"{r.name} {r.size * (not ignore_fused_rank_size)}"
                for _, r in fused_loops
            ]
            # Whether we're co-tiled with the other op. If we are co-tiled,
            # then need to keep track because it affects whether utilization is
            # additive between the operations or not
            + [str(bool(set(other.tensor_names) & set(self._co_tiled_tensors)))]
        )

    @functools.lru_cache(10000)
    def compatible_with(
        self,
        other: "FusedMapping",
        ignore_fused_ranks=False,
        ignore_fused_rank_size=True,
    ):
        my_tensors, my_fused_loops = self.get_relevant_components(
            other.op, ignore_fused_ranks
        )
        other_tensors, other_fused_loops = other.get_relevant_components(
            self.op, ignore_fused_ranks
        )

        my_fused_tensor_names = set(t.name for t in my_tensors)
        other_fused_tensor_names = set(t.name for t in other_tensors)
        if my_fused_tensor_names != other_fused_tensor_names:
            return False

        # No fused tensors! Automatically compatible
        if not my_fused_tensor_names and not other_fused_tensor_names:
            return True

        # Only working with ranks below this point
        if ignore_fused_ranks:
            return True

        # One of us is fully untiled, so the other can do whatever
        if not my_fused_loops or not other_fused_loops:
            return True

        mine = list(m[0] for m in my_fused_loops)
        other = list(o[0] for o in other_fused_loops)
        while mine and other:
            m, o = mine.pop(0), other.pop(0)
            if not m.overlaps(o):
                return False
            # BUGFIX: MAKE SURE THAT SIZE IS INCLUDED FOR ALL BUT THE
            # LAST (INNERMOST) FUSED LOOP
            # if not ignore_fused_rank_size and m[0].size != o[0].size:
            #     return False
            while mine and mine[0].overlaps(o):
                mine.pop(0)
            while other and other[0].overlaps(m):
                other.pop(0)

        if mine or other:
            return False

        return True

    def _get_str(self) -> str:
        return " ".join(
            [p.name for p in sorted(self.op, key=lambda x: x.name)]  # Unordered
            + [str(s) for s in self.fused_ranks]  # Ordered
            + sorted(t.name for t in self.fused_tensors)  # Unordered
        )

    def __str__(self) -> str:
        return self._as_str

    def __hash__(self) -> int:
        return self._hashed

    def results_match(self, other: "FusedMapping"):
        df1, df2 = self.df, other.df
        # Compare the paretos.ACCESSES_COL and paretos.UTIL_COL columns.
        if len(df1) != len(df2):
            return False

        # Sort them by the columns
        df1 = df1.sort_values([paretos.ACCESSES_COL, paretos.UTIL_COL])
        df2 = df2.sort_values([paretos.ACCESSES_COL, paretos.UTIL_COL])

        # Compare the sorted dataframes
        return df1[[paretos.ACCESSES_COL, paretos.UTIL_COL]].equals(
            df2[[paretos.ACCESSES_COL, paretos.UTIL_COL]]
        )

    @staticmethod
    def _concat_fused(mappings: Iterable["FusedMapping"], delay: bool) -> Pareto:
        p = delayed(Pareto.concat_different)(
            [m.mapping_result for m in mappings],
            utilization_additive=True,
            flatten_different_fusions=True,
        )
        return p if delay else p[0](*p[1], **p[2])

    @staticmethod
    def concat_different(mappings: List["FusedMapping"], delay: bool = False):
        if len(mappings) == 1:
            return mappings[0]

        opnames = [set(o.name for o in m.op) for m in mappings]
        for i, o in enumerate(opnames):
            for j in range(i + 1, len(opnames)):
                assert not o & opnames[j], (
                    f"Can't concat_different overlapping op names: "
                    f"{o} and {opnames[j]} overlap in {o & opnames[j]}"
                )

        # First partition the mappings into disjoint fused kernels. Utilization
        # is additive between these kernels
        disjoint_kernels = [[m] for m in mappings]
        changed = True
        while changed:
            changed = False
            for i1, m1 in enumerate(disjoint_kernels):
                for i2, m2 in enumerate(disjoint_kernels[i1 + 1 :], start=i1 + 1):
                    if next(iter(m1)).is_tiled_fused_with(next(iter(m2))):
                        disjoint_kernels[i2].extend(m1)
                        disjoint_kernels.pop(i1)
                        changed = True
                        break
                if changed:
                    break

        disjoint_paretos = []
        for d in disjoint_kernels:
            disjoint_paretos.append(
                FusedMapping._concat_fused(frozenset(d), delay=delay)
            )

        p = delayed(Pareto.concat_different)(
            disjoint_paretos,
            utilization_additive=False,
            flatten_different_fusions=True,
        )
        if not delay:
            p = p[0](*p[1], **p[2])

        fused_tensors = []
        ops = []
        fused_loops = []
        fusion_string = []
        for m in sorted(mappings, key=lambda x: str(x)):
            fused_tensors.extend([t for t in m.fused_tensors if t not in fused_tensors])
            ops.extend([o for o in m.op if o not in ops])
            fused_loops.extend(
                [
                    l
                    for l in m.fused_loops
                    if not any(l[0] == l2[0] for l2 in fused_loops)
                ]
            )
            fusion_string.append(m.fusion_string)

        new_mapping = FusedMapping(
            fused_tensors=fused_tensors,
            fused_loops=fused_loops,
            op=ops,
            fusion_string="",
            mapping_result=p,
        )
        return new_mapping

    @staticmethod
    def _concat_same(mappings: Iterable["FusedMapping"], delay: bool) -> Pareto:
        c = delayed(Pareto.concat_same)(
            [m.mapping_result for m in mappings], flatten_different_fusions=True
        )
        return c if delay else c[0](*c[1], **c[2])

    @staticmethod
    def concat_same(mappings: List["FusedMapping"], delay: bool = False):
        if len(mappings) == 1:
            return next(iter(mappings))
        mappings = list(mappings)

        new_mapping = FusedMapping(
            fused_tensors=mappings[0].fused_tensors,
            fused_loops=mappings[0].fused_loops,
            op=mappings[0].op,
            fusion_string="",
            mapping_result=FusedMapping._concat_same(frozenset(mappings), delay=delay),
        )
        return new_mapping

    def undelay_mapping_result(self) -> None:
        # Resolves if the mapping result is from a delayed() call
        self.mapping_result = self.mapping_result[0](
            *self.mapping_result[1], **self.mapping_result[2]
        )


class PotentialMultiLayerMapping(dict):
    def partition_by(self, by: Operation, connections: List[Operation]):
        partitioned = []
        connections = tuple(connections)

        # Group solutions for this operation by its connections
        cur_grouped: Dict[List[PotentialMultiLayerMapping]] = defaultdict(list)
        for d in self[by.name]:
            cur_grouped[d.strby(connections)].append(d)

        # Group solutions for the other operations by this operation
        all_conns_grouped: List[Dict[PotentialMultiLayerMapping]] = []
        for c in connections:
            all_conns_grouped.append(defaultdict(list))
            for d in self[c.name]:
                all_conns_grouped[-1][d.strby(by)].append(d)

        # For each pairing of groups, if it's compatible, add in this solution
        partitioned = []
        for cur_group in cur_grouped.values():
            possible_solutions = []
            for c, conn_grouped in zip(connections, all_conns_grouped):
                possible_solutions.append([])
                for conn_group in conn_grouped.values():
                    if cur_group[0].compatible_with(conn_group[0]):
                        possible_solutions[-1] += conn_group

            if not all(possible_solutions):
                continue

            partitioned.append(PotentialMultiLayerMapping(**self))
            for c, conns in zip(connections, possible_solutions):
                partitioned[-1][c.name] = frozenset(conns)
            partitioned[-1][by.name] = frozenset(cur_group)
        return partitioned

    @staticmethod
    def prune(
        solutions: List["PotentialMultiLayerMapping"],
        ops_visited: List[Operation],
        ops_left: List[Operation],
    ) -> List["PotentialMultiLayerMapping"]:
        all_op_names = {p.name for p in ops_left} | {p.name for p in ops_visited}
        for s in solutions:
            names = set(s.keys())
            if "dead" in s:
                names.remove("dead")
                for j, d in enumerate(s["dead"]):
                    fnames = d.mapping_result.opnames | names
                    assert (
                        fnames == all_op_names
                    ), f"{j}: {sorted(fnames)}\n {sorted(d.mapping_result.opnames)}\n {sorted(names)} != {sorted(all_op_names)}. \n\n Dropped {all_op_names - fnames}"
                names |= set(o.name for o in list(s["dead"])[0].op)
            assert names == all_op_names, f"{sorted(names)} != {sorted(all_op_names)}"

        # Now we're going to prune the new solutions the following criteria.
        # - An op is "dead" if it does not connect to any other ops outside the
        #   current solution set & it is not tiled fused with any live ops.
        # - If two solutions vary only in live ops, they can be merged into
        #   one solution. Flatten the mappings of the dead ops.

        remaining = {p.name for p in ops_left}
        live = {
            p.name for p in ops_visited if any(p.is_connected_to(p2) for p2 in ops_left)
        }
        combined = defaultdict(list)

        # NOTE: Previously, a live node was live if it is unvisited, connected to unvisited,
        # OR tiled with a live node. The third case was because we didn't want to combine
        # incomplete paretos before we calculate their co-pareto because some may be co-tiled
        # with later layers, leading in an additive utilization calculation, while others
        # may not be co-tiled, leading to a maximal utilization calculation. Now, co-tiling info
        # is included in strby for an op, so these two cases are always partitioned automatically.

        all_live = [l for l in live]

        to_concat = []
        for s in tqdm.tqdm(solutions, leave=False, desc="Picking solutions to prune"):
            # Combine the dead nodes into one co-pareto
            # You're live if you are connected to any unprocessed nodes OR are
            # additive connected to a live node & maximal connected to a non live node

            dead = set(s.keys()) - live - remaining
            to_check = [l for l in live]
            all_live = [l for l in live]
            while to_check:
                l = to_check.pop()
                tiled_fused = {
                    d
                    for d in dead
                    if next(iter(s[d])).is_tiled_fused_with(next(iter(s[l])))
                }
                to_check.extend(tiled_fused)
                all_live.extend(tiled_fused)
                dead -= tiled_fused

            if len(dead) > 1:
                deads = [s.pop(p) for p in dead]
                f = FusedMapping.concat_different(
                    [FusedMapping.concat_same(d, delay=True) for d in deads],
                    delay=True,
                )
                to_concat.append(f)
                s["dead"] = frozenset([f])

            # Combine if:
            # - same live/dead ops
            # - same permitted mappings for all live ops
            key = tuple(
                [frozenset(s.keys())] + [(p, frozenset(s[p])) for p in sorted(all_live)]
            )
            combined[key].append(s)

        for s, d in zip(
            to_concat,
            parallel_proc(
                [s.mapping_result for s in to_concat],
                pbar=f"Pruning solutions",
                leave=False,
            ),
        ):
            s.mapping_result = d

        if not live:
            return solutions

        for s in solutions:
            names = set(s.keys())
            if "dead" in s:
                names.remove("dead")
                for j, d in enumerate(s["dead"]):
                    fnames = d.mapping_result.opnames | names
                    assert (
                        fnames == all_op_names
                    ), f"{j}: {sorted(fnames)}\n {sorted(d.mapping_result.opnames)}\n {sorted(names)} != {sorted(all_op_names)}. \n\n Dropped {all_op_names - fnames}"
                names |= set(o.name for o in list(s["dead"])[0].op)
            assert names == all_op_names, f"{sorted(names)} != {sorted(all_op_names)}"

        to_concat = []

        def concat_solutions(solution, delay=True):
            if len(solution) == 1:
                return solution[0]
            all_dead = [f for s in solution for f in s.get("dead", [])]

            new_solution = PotentialMultiLayerMapping()

            if all_dead:
                s = FusedMapping.concat_same(all_dead, delay=delay)
                new_solution["dead"] = frozenset([s])
                to_concat.append(s)

            for k in solution[0]:
                if k == "dead":
                    continue
                new_solution[k] = []
                for s in solution[1:]:
                    for s2 in s[k]:
                        if s2 not in new_solution[k]:
                            new_solution[k].append(s2)
            return new_solution

        solutions = [concat_solutions(s) for s in combined.values()]

        for s, d in zip(
            to_concat,
            parallel_proc(
                [s.mapping_result for s in to_concat],
                pbar=f"Recombining solutions",
                leave=False,
            ),
        ):
            s.mapping_result = d

        for s in solutions:
            names = set(s.keys())
            if "dead" in s:
                names.remove("dead")
                for j, d in enumerate(s["dead"]):
                    fnames = d.mapping_result.opnames | names
                    assert (
                        fnames == all_op_names
                    ), f"{j}: {sorted(fnames)}\n {sorted(d.mapping_result.opnames)}\n {sorted(names)} != {sorted(all_op_names)}. \n\n Dropped {all_op_names - fnames}"
                names |= set(o.name for o in list(s["dead"])[0].op)
            assert names == all_op_names, f"{sorted(names)} != {sorted(all_op_names)}"

        return solutions

    def filter_by_pareto(self, layer_name: Union[str, List[str]], by: Callable) -> None:
        if isinstance(layer_name, list):
            for l in layer_name:
                self.filter_by_pareto(l, by)
            return
        self[layer_name] = [s for s in self[layer_name] if by(s)]

    def filter_by_solution(
        self, layer_name: Union[str, List[str]], by: Callable
    ) -> None:
        if isinstance(layer_name, list):
            for l in layer_name:
                self.filter_by_solution(l, by)
            return
        for s in self[layer_name]:
            df = s.mapping_result.df
            df = df[df.apply(by, axis=1)].copy()
            s.mapping_result.df = df
        self[layer_name] = [s for s in self[layer_name] if len(s.mapping_result.df)]

    @staticmethod
    def get_final_fused_solutions(
        solutions: List["PotentialMultiLayerMapping"],
        op_names: Optional[Union[List[str], List[Operation]]] = None,
        pareto_only: bool = False,
    ) -> List[FusedMapping]:
        to_plot = []

        names = ["dead"] if (op_names is not None and "dead" in op_names) else []
        if op_names:
            for o in op_names:
                if isinstance(o, Operation):
                    names.append(o.name)
                names.append(o)
        else:
            names.extend(list(set.union(*[set(s.keys()) for s in solutions])))

        for sol in solutions:
            # print(sol)
            mappings = []
            for s in [n for n in names if n in sol]:
                # print(sol[s])
                mappings.append(FusedMapping.concat_same(sol[s]))
            # print(mappings)
            m1 = FusedMapping.concat_different(mappings)
            to_plot.append(m1)
        if pareto_only:
            to_plot = [FusedMapping.concat_same(to_plot)]
        return to_plot


def combine_solutions_access_cache(cachekey, data: Any = None):
    if cachekey is None:
        return None

    THIS_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
    cache_file = os.path.join(THIS_SCRIPT_DIR, f"fusion_cache/{cachekey}.pkl")
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)

    if data is None:
        assert os.path.exists(cache_file)
        if os.path.exists(cache_file):
            print(f"Loading from cached {cache_file}")
            x = pkl.load(open(cache_file, "rb"))
            return x

    pkl.dump(data, open(cache_file, "wb"))
    print(f"Saved to cache {cache_file}.")


def combine_solutions(
    solutions: List[PotentialMultiLayerMapping],
    operations: List[Operation],
    must_fuse_easily_fusable: bool = True,
    enable_fusion: bool = True,
    plot_initial: bool = False,
    plot_after_partition: bool = False,
    plot_after_prune: bool = False,
    cachekey: str = None,
    force_rerun: bool = False,
) -> PotentialMultiLayerMapping:
    # ====================================================================
    # Step -1: See if we have cached this result
    # ====================================================================
    if not force_rerun:
        cached = combine_solutions_access_cache(cachekey)
        if cached is not None:
            return cached

    solutions = [PotentialMultiLayerMapping(**v) for v in solutions]

    # ====================================================================
    # Step 0: Pick what tensors we must / must not fuse
    # ====================================================================
    must_fuse_tensors = set()
    must_not_fuse_tensors = set(t.name for t in operations.tensors if not t.is_fusable)
    if must_fuse_easily_fusable:
        # Easily fusable IF:
        # 1. There are 2 or fewer tensors
        # 2. OR there are >2 tensors and the smallest
        #    is >128x smaller than the next smallest
        for op in operations:
            in_tensors = set(t.name for t in op.input_tensors)
            out_tensors = set(t.name for t in op.output_tensors)
            tensor_sizes = sorted(t.size() for t in op.tensors)
            if len(op.input_tensors) <= 1:
                must_fuse_tensors |= in_tensors
            elif tensor_sizes[0] * 128 < tensor_sizes[1]:
                must_fuse_tensors |= in_tensors

        all_fused_tensors = set()
        for s in solutions:
            for v in s.values():
                for f in v:
                    all_fused_tensors |= set(f.fused_tensor_names)

        must_fuse_tensors &= all_fused_tensors
        must_fuse_tensors -= must_not_fuse_tensors

        for s in solutions:
            for k, v in s.items():
                op = operations.get_op_by_name(k)
                tnames = set(t.name for t in op.tensors) & must_fuse_tensors
                # Only keep solutions that fuse all of tnames
                s[k] = [f for f in v if not (tnames - set(f.fused_tensor_names))]

    if not enable_fusion:
        for s in solutions:
            for k, v in s.items():
                s[k] = [
                    f for f in v if not (set(f.fused_tensor_names) - must_fuse_tensors)
                ]

    print(f"Must fuse tensors: {must_fuse_tensors}")
    print(f"Must not fuse tensors: {must_not_fuse_tensors}")

    ops_left = OperationList(operations)
    ops_visited = OperationList([ops_left.pop()])
    op = operations.get_op_by_name(ops_visited[-1].name)
    baseline_util, baseline_accesses = 0, 0
    if op.baseline_utilization is not None:
        baseline_util = max(baseline_util, op.baseline_utilization)
    if op.baseline_accesses is not None:
        baseline_accesses += op.baseline_accesses

    if plot_initial:
        for k in solutions[0]:
            s = PotentialMultiLayerMapping.get_final_fused_solutions(solutions, [k])
            op = operations.get_op_by_name(k)
            plots.plot(
                s,
                f"Initial solutions for op {k}",
                [(op.baseline_utilization, op.baseline_accesses)],
            )

    t_ops_left = tqdm.tqdm(total=len(ops_left))
    while ops_left:
        options = []

        # ====================================================================
        # STEP 1: Select a new node to add to the graph of processed nodes.
        # ====================================================================

        # Heuristic for speed: Try to close open loops in our operations graph
        # More open loops -> more potential solutions to track
        LOOP_CLOSURE_RANGE = 5
        for op in ops_left:
            connections = [c for c in ops_visited if op.is_connected_to(c)]
            if not connections:
                continue
            search = [op]
            score = []
            for _ in range(LOOP_CLOSURE_RANGE):
                score.append(
                    len(
                        [
                            c
                            for c in search
                            if any(c.is_connected_to(c2) for c2 in ops_visited)
                        ]
                    )
                )
                search = [
                    c for c in operations if any(c.is_connected_to(c2) for c2 in search)
                ]
            # If we can't close a loop, try not to open loops. Pick ops with fewer neighbors.
            score.append(-len([c for c in operations if c.is_connected_to(op)]))
            options.append((op, connections, tuple(score)))
        if not options:
            raise ValueError("Disconnected graph.")

        op, connections, _ = max(options, key=lambda x: (x[-1]))
        ops_left.remove(op)

        assert len(
            solutions
        ), f"Failed before partitioning. Ops visited: {[o.name for o in ops_visited]}"

        t_ops_left.set_description(
            f"Processing {op.name} with {len(solutions)} previous solutions"
        )

        new_solutions = []

        # ====================================================================
        # STEP 2: Partition solutions. Divide up existing solutions based on
        # how they may be fused with this new node. Divide up solutions to
        # the current node based on how they may be fused with the set of
        # processed nodes. Get every valid combination of the two.
        # ====================================================================
        def partition_by(s: PotentialMultiLayerMapping, op: Operation, connections):
            return s.partition_by(op, connections)

        new_solutions = serial(
            [delayed(partition_by)(s, op, connections) for s in solutions],
            pbar=f"Generating solutions",
            leave=False,
        )
        solutions = [s for s2 in new_solutions for s in s2]
        ops_visited.append(op)

        assert len(
            solutions
        ), f"Failed after partitioning. Ops visited: {[o.name for o in ops_visited]}"

        if op.baseline_utilization is not None:
            baseline_util = max(baseline_util, op.baseline_utilization)
        if op.baseline_accesses is not None:
            baseline_accesses += op.baseline_accesses

        if plot_after_partition:
            s = PotentialMultiLayerMapping.get_final_fused_solutions(
                solutions, ops_visited
            )
            plots.plot(
                s,
                f"After partitioning by op {len(ops_visited)} {op.name}",
                [(baseline_util, baseline_accesses)],
            )

        # ====================================================================
        # STEP 3: Prune solutions. For all the valid combinations of
        # solutions, if the choice between solutions A and B will have no
        # impact on our future choices, then we can pick the pareto-optimal
        # of A and B.
        # ====================================================================
        solutions = PotentialMultiLayerMapping.prune(solutions, ops_visited, ops_left)
        assert len(
            solutions
        ), f"Failed after pruning. Ops visited: {[o.name for o in ops_visited]}"
        t_ops_left.update()

        if plot_after_prune:
            s = PotentialMultiLayerMapping.get_final_fused_solutions(
                solutions, ops_visited
            )
            plots.plot(
                s,
                f"After partitioning & pruning by op {len(ops_visited)} {op.name}",
                [(baseline_util, baseline_accesses)],
            )

    # ====================================================================
    # Step N: Write result to cache.
    # ====================================================================
    combine_solutions_access_cache(
        cachekey, (operations, solutions, baseline_util, baseline_accesses)
    )

    return operations, solutions, baseline_util, baseline_accesses
