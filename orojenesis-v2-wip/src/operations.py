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

from collections import defaultdict
import copy
import functools
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

import pandas as pd

from tensors import Rank, Tensor
import pydot
from collections import defaultdict
import math
import pydot
from typing import Iterable, List, Dict, Optional, Set, Tuple, Union
from string import ascii_uppercase
from functools import cached_property
import paretos
import struct
from util import *


class SharedRank:
    def __init__(
        self,
        from_ranks: Union[Rank, List[Rank]],
        to_ranks: Union[Rank, List[Rank]],
    ):
        makelist = lambda x: x if isinstance(x, list) else [x]
        self._from_ranks: List[Rank] = makelist(from_ranks)
        self._to_ranks: List[Rank] = makelist(to_ranks)

    @cached_property
    def tensors(self):
        tensors = []
        for r in self.ranks:
            if r.tensor not in tensors:
                tensors.append(r.tensor)
        return tensors

    @property
    def from_ranks(self):
        return self._from_ranks

    @property
    def to_ranks(self):
        return self._to_ranks

    @property
    def ranks(self):
        return self.from_ranks + self.to_ranks

    def get_bounds(self):
        tensor2ranks_in, tensor2ranks_out = self.get_tensor2ranks()
        tensor2size_in = {
            k: math.prod(r.size for r in v) for k, v in tensor2ranks_in.items()
        }
        tensor2size_out = {
            k: math.prod(r.size for r in v) for k, v in tensor2ranks_out.items()
        }
        return tensor2size_in, tensor2size_out

    def __str__(self) -> str:
        return f"{self.from_ranks}->{self.to_ranks}"

    def __repr__(self):
        return f"SharedRank(from_ranks={self.from_ranks}, to_ranks={self.to_ranks})"

    def get_tensor2ranks(self) -> Tuple[Dict[str, List[Rank]], Dict[str, List[Rank]]]:
        tensor2ranks_in, tensor2ranks_out = defaultdict(list), defaultdict(list)
        for r in self.from_ranks:
            tensor2ranks_in.setdefault(r.tensor.name, []).append(copy.copy(r))
        for r in self.to_ranks:
            tensor2ranks_out.setdefault(r.tensor.name, []).append(copy.copy(r))

        def assert_same(ranks0, ranks1):
            ranks0 = sorted(ranks0, key=lambda x: x.name)
            ranks1 = sorted(ranks1, key=lambda x: x.name)
            if len(ranks0) != len(ranks1) or not all(
                r0.size == r1.size for r0, r1 in zip(ranks0, ranks1)
            ):
                raise ValueError(
                    f"For multiple input tensors or multiple output tensors, ranks "
                    f"must match in {self}. Got a shared_rank with {ranks0} that does not "
                    f"match {ranks1}."
                )

        def pad_to_match(ranks0, ranks1):
            ranks0 = sorted(ranks0, key=lambda x: x.name)
            ranks1 = sorted(ranks1, key=lambda x: x.name)

            if len(ranks0) != len(ranks1):
                raise ValueError(
                    f"Number of ranks in {self} does not match across tensors: "
                    f"{ranks0} != {ranks1}"
                )

            for i in range(len(ranks0)):
                ranks0[i].size = max(ranks0[i].size, ranks1[i].size)
                ranks1[i].size = ranks0[i].size

        # Assert that all input ranks are the same
        # for r0 in tensor2ranks_in.values():
        #     for r1 in tensor2ranks_in.values():
        #         assert_same(r0, r1)
        # for r0 in tensor2ranks_out.values():
        #     for r1 in tensor2ranks_out.values():
        #         assert_same(r0, r1)

        # Instead of doing the above, which complained about some reshapes dropping
        # and/or adding data, we will just assume everything is padded to the max
        # size
        for r0 in tensor2ranks_in.values():
            for r1 in tensor2ranks_in.values():
                pad_to_match(r0, r1)
        for r0 in tensor2ranks_out.values():
            for r1 in tensor2ranks_out.values():
                pad_to_match(r0, r1)

        return tensor2ranks_in, tensor2ranks_out

    def to_rank(self):
        tensor2ranks_in, tensor2ranks_out = self.get_tensor2ranks()
        if tensor2ranks_in:
            ranks = next(iter(tensor2ranks_in.values()))
            if len(ranks) == 1:
                return ranks[0]
        if tensor2ranks_out:
            ranks = next(iter(tensor2ranks_out.values()))
            if len(ranks) == 1:
                return ranks[0]

        raise ValueError(
            "Cannot convert shared_rank to a single rank when there are "
            "multiple shared_ranked or zero shared_ranked ranks on both sides."
        )

    @property
    def name(self) -> str:
        return " ".join(sorted(str(s) for s in self.ranks))

    def overlaps(self, other: Union["SharedRank", Tensor]) -> bool:
        if isinstance(other, SharedRank):
            return any(
                r1.is_equivalent_to(r2) for r1 in self.ranks for r2 in other.ranks
            )
        if isinstance(other, Tensor):
            return any(r.tensor.name == other.name for r in self.ranks)
        raise TypeError(f"Can not calculate overlaps for type {type(other)}")


class Operation:
    def __init__(
        self,
        name: str,
        input_tensors: Iterable[Tensor],
        output_tensors: Iterable[Tensor],
        # Absent from none: G
        # Absent from inputs: N
        # Absent from outputs: K
        # Absent from weights: M
        absent_ranks: dict[str:Rank] = None,
        baseline_accesses: int = None,
        baseline_utilization: int = None,
    ):
        self.name = name
        self.input_tensors: List[Tensor] = list(input_tensors)
        self.output_tensors: List[Tensor] = list(output_tensors)

        assert len(set(t.name for t in self.input_tensors)) == len(
            self.input_tensors
        ), f"Duplicate tensors in op {name}"
        assert len(set(t.name for t in self.output_tensors)) == len(
            self.output_tensors
        ), f"Duplicate tensors in op {name}"

        self.absent_ranks: Dict[str : Set[Rank]] = absent_ranks or {}
        self.shared_ranks: List[SharedRank] = []
        self.baseline_accesses: int = baseline_accesses
        self.baseline_utilization: int = baseline_utilization
        self.force_zero_accesses: bool = False
        self.scale_accesses_by: int = 1

    def add_shared_rank(self, shared_rank: SharedRank):
        self.shared_ranks.append(shared_rank)

    def update_used_in_ops(self):
        for tensor in self.tensors:
            tensor.used_in_ops.add(self)
        for tensor in self.output_tensors:
            tensor.output_of_ops.add(self)

    def __repr__(self) -> str:
        return f"Operation({repr(self.name)}, input_tensors=[{','.join([repr(t) for t in self.input_tensors])}], output_tensors=[{','.join([repr(t) for t in self.output_tensors])}"

    def add_absent_rank(self, absent_from: str, rank: Rank) -> None:
        self.absent_ranks.setdefault(absent_from, set()).add(rank)

    def get_shared_ranks(
        self, tensor: Optional[Tensor] = None, rank: Optional[Rank] = None
    ) -> List[SharedRank]:
        shared_ranks = self.shared_ranks
        if tensor is not None:
            shared_ranks = [
                l for l in shared_ranks if tensor in l.from_tensors + l.to_tensors
            ]
        if rank is not None:
            shared_ranks = [
                l for l in shared_ranks if rank in l.from_ranks + l.to_ranks
            ]
        return shared_ranks

    def auto_link_shared_ranks(self, squash_extras: bool = False) -> None:
        # Auto shared_rank: Connect ranks with equal factors. Error on >1 or <1 factors per rank.
        # Manual shared_rank:
        #  - One input tensor to multiple output tensors: Assert sum-of-products equal. We're partitioning the tensor.
        #  - Multiple input tensors to one output tensor: Assert sum-of-products equal. We're concatenating the tensors.
        #  - One tensor to one tensor:
        #    - Products equal: Simple reshape
        #    - Output product >: We're fetching new data. KV cache.
        #    - Output product <: We're dropping data. Backpropagate lowered fusion requirements.
        #
        # First, track all tensors and ranks
        input_targets = [r for t in self.input_tensors for r in t.ranks]
        output_targets = [r for t in self.output_tensors for r in t.ranks]

        from_ranks_with_no_to_ranks = []

        def drop_by_shared_rank(shared_rank: SharedRank):
            nonlocal input_targets
            nonlocal output_targets
            input_targets = [
                r for r in input_targets if r not in shared_rank.from_ranks
            ]
            output_targets = [
                r for r in output_targets if r not in shared_rank.to_ranks
            ]

        # Don't count ranks that have already been shared_ranked manually
        for l in self.shared_ranks:
            # print(f'{self.name} dropping by {l}')
            drop_by_shared_rank(l)

        def errmsg(err: str):
            raise ValueError(
                f"Could not automatically shared_rank ranks for {self}. {err} "
                f"Try manually sharing ranks. Auto shared_ranks found:\n\t"
                + "\n\t".join(str(l) for l in self.shared_ranks)
            )

        # Now we have a list of input and output targets that are not shared_ranked to any shared_rank.
        while input_targets:
            rank = input_targets.pop()

            from_ranks = [rank]
            to_ranks = []

            for ranks, targets in [
                (from_ranks, input_targets),
                (to_ranks, output_targets),
            ]:
                for rank2 in targets:
                    if rank.tensor == rank2.tensor:
                        continue
                    if rank.size == rank2.size:
                        ranks.append(rank2)

            # Now assert that each tensor shows up at least once
            tensor2rank = {}
            for r in from_ranks + to_ranks:
                # If it hasn't been seen yet, OK
                if r.tensor.name not in tensor2rank:
                    tensor2rank[r.tensor.name] = r
                    continue

                other = tensor2rank[r.tensor.name]

                # Give preference to one if the names match
                if rank.name not in [r.name, other.name]:
                    errmsg(
                        f"Aliased size for rank {rank} -> ({r}, {other}) for {r.tensor.name}"
                    )

                if r.name == rank.name:
                    to_drop = other
                    tensor2rank[r.tensor.name] = r
                else:
                    to_drop = r

                from_ranks = [f for f in from_ranks if f != to_drop]
                to_ranks = [t for t in to_ranks if t != to_drop]

            num_shared_ranked_ranks = len(from_ranks) + len(to_ranks)
            num_tensors = len(self.input_tensors) + len(self.output_tensors)
            if num_shared_ranked_ranks != 2 and num_shared_ranked_ranks != num_tensors:
                if squash_extras and num_shared_ranked_ranks == 1:
                    from_ranks_with_no_to_ranks += from_ranks
                    continue
                else:
                    errmsg(
                        f"Rank {rank} must be shared_ranked between TWO or ALL tensors, not {num_shared_ranked_ranks}."
                    )

            self.add_shared_rank(SharedRank(from_ranks, to_ranks))
            drop_by_shared_rank(self.shared_ranks[-1])

        if squash_extras:
            if from_ranks_with_no_to_ranks and output_targets:
                self.add_shared_rank(
                    SharedRank(from_ranks_with_no_to_ranks, output_targets)
                )
                drop_by_shared_rank(self.shared_ranks[-1])
            elif output_targets:
                # Find a shared_rank with no targets
                shared_ranks_with_no_targets = [
                    l for l in self.shared_ranks if len(l.to_ranks) == 0
                ]
                if len(shared_ranks_with_no_targets) > 1:
                    errmsg(
                        f"Must squash output targets {output_targets} into one shared_rank, "
                        f"but got {len(shared_ranks_with_no_targets)} shared_ranks with no target."
                    )
                if len(shared_ranks_with_no_targets) == 1:
                    l = shared_ranks_with_no_targets[0]
                    new_l = SharedRank(l.from_ranks, output_targets)
                    self.shared_ranks[self.shared_ranks.index(l)] = new_l
                    drop_by_shared_rank(new_l)
                else:
                    self.add_shared_rank(SharedRank([], output_targets))
                    drop_by_shared_rank(self.shared_ranks[-1])

            elif from_ranks_with_no_to_ranks:
                self.add_shared_rank(SharedRank(from_ranks_with_no_to_ranks, []))
                drop_by_shared_rank(self.shared_ranks[-1])

        if input_targets or output_targets:
            errmsg(f"Could not share ranks{input_targets + output_targets}")

    def resolve_ranks(self):
        for rank in set(r for t in self.tensors for r in t.ranks):
            absent_from = [t for t in self.tensors if rank not in t.ranks]
            if len(absent_from) == 0:
                self.add_absent_rank(None, rank)
            elif len(absent_from) == 1:
                self.add_absent_rank(absent_from[0].name, rank)
            else:
                raise ValueError(
                    f"Rank {rank} is absent from multiple tensors: {[t.name for t in absent_from]}"
                )

    @property
    def tensors(self) -> List[Tensor]:
        return self.input_tensors + self.output_tensors

    @property
    def ranks(self) -> List[Rank]:
        return [rank for tensor in self.tensors for rank in tensor.ranks]

    @cached_property
    def tensor_names(self) -> List[str]:
        return [t.name for t in self.tensors]

    def shared_rank_with_rank(self, rank: Rank) -> SharedRank:
        shared_ranks = [l for l in self.shared_ranks if rank in l.ranks]
        if len(shared_ranks) != 1:
            raise ValueError(
                f"Expected exactly one shared_rank with rank {rank}, got {len(shared_ranks)}"
            )
        return shared_ranks[0]

    def to_timeloop_dict(self) -> Dict[str, int]:
        dims = defaultdict(lambda: 1)

        taken_names = set()

        def get_unique_name(base):
            if base not in taken_names:
                taken_names.add(base)
                return base
            i = 2
            while f"{base}{i}" in taken_names:
                i += 1
            taken_names.add(f"{base}{i}")
            return f"{base}{i}"

        ranknames = {
            "Inputs|Outputs": "M",
            "Inputs|Outputs|Outputs2|Outputs3": "M",
            "Inputs|Weights": "K",
            "Inputs|Inputs2": "K",
            "Outputs|Weights": "N",
            "Inputs2|Outputs": "N",
            "Inputs|Outputs|Weights": "G",
            "Inputs|Inputs2|Outputs": "G",
            "Outputs": "T",
        }

        for t in self.input_tensors:
            if "weight" in t.name:
                t._brief_name = get_unique_name("Weights")
            else:
                t._brief_name = get_unique_name("Inputs")
        for t in self.output_tensors:
            t._brief_name = get_unique_name("Outputs")

        for l in self.shared_ranks:
            rankname = "|".join(sorted([t._brief_name for t in l.tensors]))
            input_bounds, ouput_bounds = l.get_bounds()
            min_input_bound = (
                min(input_bounds.values()) if input_bounds else float("inf")
            )
            min_output_bound = (
                min(ouput_bounds.values()) if ouput_bounds else float("inf")
            )
            result = min(min_input_bound, min_output_bound)
            if result > 1:
                dims[ranknames[rankname]] *= result

        required_keys = "MNKGT"
        for k in required_keys:
            dims.setdefault(k, 1)
        required_keys = "MNKGT"
        assert len(dims) == len(required_keys), f"{dims} {required_keys}"

        return dict(dims)

    def is_dependent_on(self, other: Union[Tensor, "Operation"]) -> bool:
        if isinstance(other, Tensor):
            return other in self.input_tensors
        if isinstance(other, Operation):
            return any(t in other.output_tensors for t in self.input_tensors)
        raise TypeError("is_dependent_on expects a Tensor or Operation")

    def is_connected_to(self, other: Union[Tensor, "Operation"]) -> bool:
        if self.is_dependent_on(other) or other.is_dependent_on(self):
            return True
        if isinstance(self, Operation) and isinstance(other, Operation):
            my_tensors = self.tensors
            other_tensors = other.tensors
            return any(t in other_tensors for t in my_tensors)

    def rename_ranks(self):
        taken_names = set()

        def get_unique_name(shared_rank):
            ranks = [r for r in shared_rank.from_ranks if r.name not in taken_names]
            if len(ranks) == 0:
                chosen = [c for c in ascii_uppercase if c not in taken_names][0]
            else:
                # Penalize name changes to encourage convergence
                chosen = min(
                    ranks, key=lambda x: len(x.tensor.name) * 1000 + len(x.name)
                ).name
            taken_names.add(chosen)
            return chosen

        for l in self.shared_ranks:
            ranks = l.ranks
            name = get_unique_name(l)
            for rank in ranks:
                rank.name = name

    def get_reduced_ranks(self) -> List[SharedRank]:
        return [
            r
            for r in self.shared_ranks
            if not any(r.overlaps(t) for t in self.output_tensors)
        ]


class OperationList(list):
    def resolve_aliased_tensors(self):
        name2tensor = {}

        def get_aliased_tensor(t: Tensor):
            if t.name in name2tensor:
                if t != name2tensor[t.name]:
                    raise ValueError(
                        f"Found differing tensors with the same name {t.name}: {t} and {name2tensor[t.name]} {t == name2tensor[t.name]}"
                    )
                return name2tensor[t.name]
            name2tensor[t.name] = t
            return t

        for op in self:
            op.input_tensors = [get_aliased_tensor(t) for t in op.input_tensors]
            op.output_tensors = [get_aliased_tensor(t) for t in op.output_tensors]

        self.update_used_in_ops()

    def get_op_by_name(
        self, name: Union[str, List[str]], subset: bool = False
    ) -> Union[Operation, "OperationList"]:
        if subset:
            if isinstance(name, str):
                name = [name]
            return OperationList(
                filter(lambda x: any([n in x.name for n in name]), self)
            )

        if isinstance(name, list):
            return OperationList(self.get_op_by_name(n) for n in name)

        for op in self:
            if op.name == name.rsplit(" ", 1)[0]:
                return op
        raise KeyError(f"No operation with name {name} found")

    def get_tensor_by_name(
        self, name: Union[str, List[str]], subset: bool = False
    ) -> Union[Tensor, List[Tensor]]:
        if subset:
            if isinstance(name, str):
                name = [name]
            return filter(lambda x: any([n in x.name for n in name]), self.tensors)

        if isinstance(name, list):
            return [self.get_tensor_by_name(n) for n in name]

        for t in self.tensors:
            if t.name == name:
                return t
        raise KeyError(f"No tensor with name {name} found")

    @property
    def tensors(self) -> List[Tensor]:
        return [t for op in self for t in op.tensors]

    def update_used_in_ops(self):
        for op in self:
            op.update_used_in_ops()

    def to_pydot(self):
        graph = pydot.Dot(graph_type="digraph", ranksep="0.25", nodesep="0.25")

        def get_tensor_node(tensor):
            ranktexts = [f"{str(rank)}" for rank in tensor.ranks]
            text = "\n".join([tensor.name] + ranktexts)
            node = pydot.Node(
                name=f"Tensor {tensor.name}".replace(":", "_").replace("/", "_"),
                shape="box",
                style="filled",
                fillcolor="#f9f9f9",
                label=text,
                fontsize="10",
                width=0,
                penwidth=1,
                margin="0",
                height=0,
            )
            graph.add_node(node)
            tensor._graph_node = node
            return node

        def get_op_node(op):
            shared_ranktexts = [
                f"({','.join(str(r) for r in l.from_ranks)})-({','.join(str(r) for r in l.to_ranks)})"
                for l in op.shared_ranks
            ]
            text = "\n".join(
                [f"{op.name}  (x{op.scale_accesses_by})"] + shared_ranktexts
            )
            # text += f"\n{op.to_timeloop_dict()}"
            node = pydot.Node(
                f"Operation {op.name}".replace(":", "_").replace("/", "_"),
                shape="box",
                style="filled",
                fillcolor="#ffcccc",
                label=text,
                fontsize="10",
                width=0,
                penwidth=1,
                margin="0",
                height=0,
            )
            graph.add_node(node)
            op._graph_node = node
            return node

        # Create nodes for each tensor and add them to the graph
        for tensor in self.tensors:
            get_tensor_node(tensor)

        for op in self:
            get_op_node(op)
            for t in op.input_tensors:
                edge = pydot.Edge(t._graph_node, op._graph_node, color="blue")
                graph.add_edge(edge)
            for t in op.output_tensors:
                edge = pydot.Edge(op._graph_node, t._graph_node, color="red")
                graph.add_edge(edge)
        return graph

    def rename_ranks(self):
        for op in self:
            op.rename_ranks()

    def to_pydot_acc_util(self, title: str, info_text: pd.DataFrame):
        title = f"{title}\n" + pretty_sci_notation(
            f"{info_text[paretos.UTIL_COL]:.2e} utilization (LEFT), "
            f"{info_text[paretos.ACCESSES_COL]:.2e} accesses (RIGHT)"
        )

        def getstat(s):
            return info_text.get(s, 0)

        max_op_util = max(getstat(f"{op.name} {paretos.UTIL_COL}") for op in self)
        max_op_acc = max(getstat(f"{op.name} {paretos.ACCESSES_COL}") for op in self)
        max_t_util = max(getstat(f"{t.name} {paretos.UTIL_COL}") for t in self.tensors)
        max_t_acc = max(
            getstat(f"{t.name} {paretos.ACCESSES_COL}") for t in self.tensors
        )

        def get_tensor_node(tensor, acc_info: bool):
            acc = getstat(f"{tensor.name} {paretos.ACCESSES_COL}")
            util = getstat(f"{tensor.name} {paretos.UTIL_COL}")
            tsize = tensor.size()

            if acc_info:
                s, t = acc / max(max_t_acc, 1), f"{acc:.1e}A (x{acc//tsize})"
            else:
                s, t = (
                    util / max(max_t_util, 1),
                    f"{util:.1e}U (/{tsize//max(util, 1)})",
                )

            t = pretty_sci_notation(t)

            importance = 0.3 + 0.7 * s
            basecolor = struct.unpack("BBB", bytes.fromhex("f9f9f9"))
            scaled = (int(i * importance) for i in basecolor)

            text = f"{tensor.name}\n{t}"
            node = pydot.Node(
                name=f"Tensor {tensor.name} {acc_info}".replace(":", "_").replace(
                    "/", "_"
                ),
                shape="box",
                style="filled",
                fillcolor="#" + bytes.hex(struct.pack("BBB", *scaled)),
                label=text,
                fontsize="10",
                width=0,
                penwidth=1,
                margin="0",
                height=0,
            )
            graph.add_node(node)
            tensor._graph_node = node
            return node

        def get_op_node(op, acc_info: bool):
            acc = getstat(f"{op.name} {paretos.ACCESSES_COL}")
            util = getstat(f"{op.name} {paretos.UTIL_COL}")

            if acc_info:
                s, t = acc / max(max_op_acc, 1), f"{acc:.1e}A"
            else:
                s, t = util / max(max_op_util, 1), f"{util:.1e}U"
            t = pretty_sci_notation(t)

            importance = 0.3 + 0.7 * s
            basecolor = struct.unpack("BBB", bytes.fromhex("ffcccc"))
            scaled = (int(i * importance) for i in basecolor)

            text = f"{op.name} (x{op.scale_accesses_by})\n{t}"
            node = pydot.Node(
                f"{op.name} {acc_info}".replace(":", "_").replace("/", "_"),
                shape="box",
                style="filled",
                fillcolor="#" + bytes.hex(struct.pack("BBB", *scaled)),
                label=text,
                fontsize="10",
                width=0,
                margin="0",
                height=0,
            )
            graph.add_node(node)
            op._graph_node = node
            return node

        graph = pydot.Dot(
            graph_type="digraph",
            label=title,
            ranksep="0.1",
            nodesep="0.1",
            labelloc="t",
        )

        # Create nodes for each tensor and add them to the graph
        for tensor in self.tensors:
            get_tensor_node(tensor, False)
        for op in self:
            get_op_node(op, False)
            for t in op.input_tensors:
                edge = pydot.Edge(t._graph_node, op._graph_node, color="blue")
                graph.add_edge(edge)
            for t in op.output_tensors:
                edge = pydot.Edge(op._graph_node, t._graph_node, color="red")
                graph.add_edge(edge)

        for tensor in self.tensors:
            get_tensor_node(tensor, True)
        for op in self:
            get_op_node(op, True)
            for t in op.input_tensors:
                edge = pydot.Edge(t._graph_node, op._graph_node, color="blue")
                graph.add_edge(edge)
            for t in op.output_tensors:
                edge = pydot.Edge(op._graph_node, t._graph_node, color="red")
                graph.add_edge(edge)

        return graph

    def rename_ranks(self):
        for op in self:
            op.rename_ranks()
