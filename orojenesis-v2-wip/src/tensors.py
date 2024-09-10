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

from typing import List, Optional, Set
from math import prod


class Rank:
    def __init__(self, name: str, size: int):
        self.name: str = name
        self.size: int = size
        self.tensor: Tensor = None

    def __str__(self) -> str:
        return f"{self.name}({self.size})"

    def is_equivalent_to(self, other):
        if isinstance(other, Rank):
            return self.name == other.name and self.size == other.size
        return False

    def __hash__(self):
        return hash((self.name, self.size))

    def __repr__(self):
        return f"Rank(name={self.name}, size={self.size})"


class Tensor:
    def __init__(self, name: str, ranks: Set[Rank], is_fusable: bool = False):
        self.name = name
        self.ranks = set(ranks)
        self.precision = None
        self.accum_precision = None
        self.is_fusable = is_fusable

        self.used_in_ops = set()
        self.output_of_ops = set()
        for r in self.ranks:
            r.tensor = self

    def get_rank_by_name(self, name: str) -> Optional[Rank]:
        for rank in self.ranks:
            if rank.name == name:
                return rank
        raise KeyError(
            f"Rank with name '{name}' not found in tensor '{self.name}' with ranks {self.ranks}"
        )

    def add_rank(self, rank: Rank):
        self.ranks.add(rank)
        rank.tensor = self

    def __str__(self) -> str:
        return f"{self.name}: {', '.join(map(str, self.ranks))}"

    def sorted_ranks(self) -> List[Rank]:
        return sorted(self.ranks, key=lambda rank: rank.name)

    def set_precision(self, precision: int):
        if self.precision is not None and precision != self.precision:
            raise ValueError(
                "Precision has already been set for tensor {}."
                "Cannot change from {} to {}".format(
                    self.name,
                    self.precision,
                    precision,
                )
            )
        self.precision = precision

    def set_accum_precision(self, precision: int):
        if self.accum_precision is not None and precision != self.accum_precision:
            raise ValueError(
                "Accumulation precision has already been set for tensor {}".format(
                    self.name
                )
            )
        self.accum_precision = precision

    def assert_precision_set(self):
        assert (
            self.precision is not None
        ), "Precision has not been set for tensor {}".format(self.name)
        # assert (
        #     self.accum_precision is not None
        # ), "Accumulation precision has not been set for tensor {}".format(self.name)

    def __eq__(self, other) -> bool:
        if self.name != other.name:
            return False
        if len(self.ranks) != len(other.ranks):
            return False
        for r0, r1 in zip(self.sorted_ranks(), other.sorted_ranks()):
            if not r0.is_equivalent_to(r1):
                return False
        return True

    def __repr__(self) -> str:
        return f"Tensor({repr(self.name)}, {', '.join(map(repr, self.ranks))})"

    def size(self) -> int:
        return prod([rank.size for rank in self.ranks]) * self.precision
