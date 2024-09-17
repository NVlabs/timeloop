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

import pandas as pd
from typing import List, Optional, Set
from util import *

ACCESSES_COL = "Total Accesses"
UTIL_COL = "Total Utilization"
TILING_COL = "Tiling"
FUSION_COL = f"Fused {TILING_COL}"
FUSION_STR_COL = f"Fused {TILING_COL} String"
FLIPPED_TC_COL = f"Flipped Tensor Core"
MAPPING_COL = f"Mapping"
FUSED_TENSOR_COL = "Fused Tensors"

MIN_UTIL_IMPROVEMENT_WORTH_HIGHER_ACCESSES = 0.95

ALL_COLS = [
    ACCESSES_COL,
    UTIL_COL,
    TILING_COL,
    FUSION_COL,
    FUSION_STR_COL,
    FLIPPED_TC_COL,
    MAPPING_COL,
    FUSED_TENSOR_COL,
]


class Pareto:
    def __init__(
        self,
        opnames: Set[str],
        df: pd.DataFrame,
        flatten_different_fusions: bool,
        trim_as_subproc: bool = False,
    ):
        self.opnames = opnames
        if not isinstance(opnames, set):
            raise TypeError("Op names must be a set")
        self.df = df
        self.flatten_different_fusions = flatten_different_fusions
        self.trim_to_optimal(as_subproc=trim_as_subproc)

    def trim_to_optimal(self, as_subproc: bool = False):
        p = delayed(Pareto.trim_optimal_df)(self.df, self.flatten_different_fusions)
        if not as_subproc:
            self.df = p[0](*p[1], **p[2])
        else:
            self.df = as_subproc(p)

    @staticmethod
    def trim_optimal_df(df: pd.DataFrame, flatten_different_fusions: bool):
        sortkeys = [ACCESSES_COL, UTIL_COL]

        if not flatten_different_fusions:
            sortkeys = [FUSION_STR_COL] + sortkeys

        if not flatten_different_fusions:
            if FUSION_STR_COL not in df.columns:
                raise KeyError(
                    f"Must have the {FUSION_STR_COL} column in df "
                    f"or have flatten_different_fusions=True."
                )

        df = df.sort_values(by=sortkeys).reset_index(drop=True)
        prevrow = df.iloc[0]
        best_util_at_this_num_accesses = prevrow[UTIL_COL]

        pareto = [True] * len(df)
        # We're already sorted in increasing accesses, so skip those checks.
        for i, row in enumerate(range(len(df))):
            row = df.iloc[i]

            utilization = row[UTIL_COL]
            newmap = row[FUSION_STR_COL] != prevrow[FUSION_STR_COL]
            betterutil = (
                utilization
                < best_util_at_this_num_accesses
                * MIN_UTIL_IMPROVEMENT_WORTH_HIGHER_ACCESSES
            )

            keep = betterutil or (newmap and not flatten_different_fusions)
            if keep:
                best_util_at_this_num_accesses = utilization
            if not keep:
                pareto[i] = False

            prevrow = row

        pareto[0] = True
        df = df[pareto]
        df.reset_index(drop=True, inplace=True)
        return df.copy()

    @staticmethod
    def concat_same(
        paretos: List["Pareto"],
        flatten_different_fusions: Optional[bool] = None,
        delay: bool = False,
    ) -> "Pareto":
        paretos = list(paretos)

        if not paretos:
            raise ValueError("Empty list of Paretos")

        names = set(frozenset(p.opnames) for p in paretos)
        if len(names) != 1:
            raise ValueError(
                f"Can't concat_same Paretos with different names:\n\t"
                + "\n\t".join(str(sorted(s)) for s in names)
            )

        d = delayed(Pareto)(
            set(paretos[0].opnames),
            pd.concat([p.df for p in paretos]),
            flatten_different_fusions,
        )
        if delay:
            return d
        return d[0](*d[1], **d[2])

    @staticmethod
    def concat_different(
        paretos: List["Pareto"],
        utilization_additive: bool,
        flatten_different_fusions: bool,
    ) -> "Pareto":
        if not paretos:
            raise ValueError("Empty list of Paretos")

        paretos = list(paretos)
        # Paretos after 2 are checked by recursive calls
        for i, p in enumerate(paretos[:2]):
            if isinstance(p, tuple):
                paretos[i] = p[0](*p[1], **p[2])
            if not isinstance(paretos[i], Pareto):
                raise TypeError("Can't concat non-Pareto objects")

        assert (
            flatten_different_fusions
        ), "Don't know how to preserve different fusions when >1 layer is involved."

        # Base cases
        if len(paretos) == 1:
            return paretos[0]
        if len(paretos) > 2:
            p = paretos[0]
            for p2 in paretos[1:]:
                p = Pareto.concat_different(
                    [p, p2],
                    utilization_additive=utilization_additive,
                    flatten_different_fusions=flatten_different_fusions,
                )
            return p

        p1, p2 = paretos[0], paretos[1]

        if p1.opnames & p2.opnames:
            raise ValueError(
                "Can't concat_different Paretos with the overlapping op names "
                f"{p1.opnames} and {p2.opnames}"
            )

        suffixes = ["_RIGHT_MERGE", "_LEFT_MERGE"]
        df = pd.merge(p1.df, p2.df, how="cross", suffixes=suffixes).copy()

        # We will have two ACCESSES_COL (one from each DataFrame), so sum them into one
        def get_merged_cols(base):
            return df[[f"{base}{suffixes[0]}", f"{base}{suffixes[1]}"]]

        df[ACCESSES_COL] = get_merged_cols(ACCESSES_COL).sum(axis=1)

        if utilization_additive:
            df[UTIL_COL] = get_merged_cols(UTIL_COL).sum(axis=1)
        else:
            df[UTIL_COL] = get_merged_cols(UTIL_COL).max(axis=1)

        # Columns that stop making sense for multiple mappings together
        for c in [FUSION_COL, FUSION_STR_COL, TILING_COL, FLIPPED_TC_COL]:
            df[c] = None

        df = Pareto.trim_optimal_df(df, flatten_different_fusions).copy()

        duplicates = [c for c in df.columns if "_RIGHT_MERGE" in c]
        for d in duplicates:
            d = d.replace("_RIGHT_MERGE", "")
            if d in ALL_COLS:
                continue
            elif "Utilization" in d:
                # Don't double count this utilization
                # CHANGED: Assumed ouble buffering --> DO double
                # count this utilization
                if False and utilization_additive:
                    df[UTIL_COL] -= get_merged_cols(d).min(axis=1)
                df[d] = get_merged_cols(d).max(axis=1)
            elif "Accesses" in d:
                df[d] = get_merged_cols(d).sum(axis=1)
            else:
                raise ValueError(
                    f"Found duplicated column {d} and don't know how to add it."
                )

        df = df[[c for c in df.columns if "_MERGE" not in c]]

        return Pareto(p1.opnames | p2.opnames, df, flatten_different_fusions)
