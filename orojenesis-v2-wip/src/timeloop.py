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
import hashlib
import math
import os
import pickle
import re
from string import ascii_uppercase
import time
from typing import Any, Callable, Dict, List, Set, Tuple
from more_itertools import powerset
from joblib import Parallel, delayed

import timeloopfe.v4 as tl
import pandas as pd

from util import *
from operations import Operation
from tensors import Rank, Tensor
from util import *
from paretos import Pareto
import paretos
from mappings import FusedMapping

# L conflicts with level names L[] in Timeloop short mapping
# XYZ are our bit dimensions
# ABCDEFGH we use for tensors
ALLOWED_RANK_NAMES = sorted(
    set(ascii_uppercase) - set("LXYZ") - set("ABCDEFGH"), reverse=True
)
ALLOWED_TENSOR_NAMES = "ABDEF"


FURTHER_TENSOR_CORE_SIZE_INCREASE = 1

TENSOR_CORE_N = 1 * FURTHER_TENSOR_CORE_SIZE_INCREASE
TENSOR_CORE_M = 1 * FURTHER_TENSOR_CORE_SIZE_INCREASE
TENSOR_CORE_K = 32  # This will be divided by the output precision

def get_tensor_core_dims(
    op: "Operation",
    tensor2ranknames: Dict[str, List[str]],
    flipped_tc: bool,
):
    tensor_core_dims = []

    k_precision = max([t.precision for t in op.input_tensors] + [4])
    tensor_core_k = TENSOR_CORE_K // k_precision

    in_ranks = [set(tensor2ranknames[t.name]) for t in op.input_tensors]
    out_ranks = [set(tensor2ranknames[t.name]) for t in op.output_tensors]

    all_out_ranks = set.union(*out_ranks)

    # Reduction ranks appear in all input tensors & no output tensors
    reduction_ranks = set.intersection(*in_ranks) - all_out_ranks
    tensor_core_dims.append((sorted(reduction_ranks), tensor_core_k))

    if len(op.input_tensors) == 1:
        return tensor_core_dims

    assert (
        len(op.input_tensors) == 2
    ), "Tensor core can only compute operations with 1 or 2 tensors"

    t0, t1 = op.input_tensors[0], op.input_tensors[1]
    if flipped_tc:
        t0, t1 = t1, t0

    ranks0 = set(tensor2ranknames[t0.name]) & all_out_ranks
    ranks1 = set(tensor2ranknames[t1.name]) & all_out_ranks

    ranks0, ranks1 = ranks0 - ranks1, ranks1 - ranks0

    tensor_core_dims.append((sorted(ranks0), TENSOR_CORE_M))
    tensor_core_dims.append((sorted(ranks1), TENSOR_CORE_N))

    return tensor_core_dims


def get_rank_name(tensor_names: str, op: Operation) -> str:
    if all(x in tensor_names for x in ["Inputs", "Outputs", "Weights"]):
        return "G"
    if all(x in tensor_names for x in ["Inputs", "Outputs"]):
        return "M"
    if all(x in tensor_names for x in ["Inputs", "Weights"]):
        return "K"
    if all(x in tensor_names for x in ["Outputs", "Weights"]):
        return "N"
    if all(x in tensor_names for x in ["Inputs"]):
        return "I"
    if all(x in tensor_names for x in ["Outputs"]):
        return "T"
    raise ValueError(f"Unknown tensor names {tensor_names} in {op}")


def get_timeloop_translation(
    tensor2ranknames: Dict[str, str], output_tensors: List[str]
):
    input_tensors = [t for t in tensor2ranknames if t not in output_tensors]
    taken_names = set()
    rank_names_remaining = list(ALLOWED_RANK_NAMES)
    translation = {}

    def get_tensor_name(preferred: str = ""):
        if preferred and preferred not in taken_names:
            taken_names.add(preferred)
            return preferred
        for letter in ALLOWED_TENSOR_NAMES:
            if letter not in taken_names:
                taken_names.add(letter)
                return letter
        raise ValueError("Ran out of tensor names")

    def get_rank_name(preferred: str = ""):
        for p in preferred:
            if p not in taken_names:
                if p in rank_names_remaining:
                    rank_names_remaining.remove(p)
                taken_names.add(p)
                return p
        return rank_names_remaining.pop()

    def setdefault_rank_name(r: str, preferred: str = ""):
        if r in translation:
            return
        translation[r] = get_rank_name(preferred)

    for t in sorted(tensor2ranknames.keys()):
        translation[t] = get_tensor_name("D" if t in output_tensors else "")

    # Shared ranks between all tensors
    tensor2ranknames_set = {k: set(v) for k, v in tensor2ranknames.items()}
    input_ranks = set.intersection(*(tensor2ranknames_set[t] for t in input_tensors))
    output_ranks = set.intersection(*(tensor2ranknames_set[t] for t in output_tensors))
    input_ranks = {i for i in input_ranks if "bits" not in input_ranks}
    output_ranks = {o for o in output_ranks if "bits" not in output_ranks}

    # Sort everything so we get deterministic output & get cache hits when we cache
    # results.
    for r in sorted(input_ranks & output_ranks):
        setdefault_rank_name(r, "GH")
    for r in sorted(input_ranks - output_ranks):
        setdefault_rank_name(r, "K")
    for r in sorted(set.union(*(tensor2ranknames_set.values()))):
        setdefault_rank_name(r, "XYZ" if "bits" in r else "MN")

    assert len(set(translation.values())) == len(
        translation
    ), f"Translation has duplicate names {translation}"

    return translation


def get_timeloop_run_dir(jinja_parse_data: Dict[str, Any]):
    jinja_parse_data = recursive_sort(jinja_parse_data)
    hashed = hashlib.md5(str(jinja_parse_data).encode("utf-8")).hexdigest()[:20]
    base_dir = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(base_dir, "outputs", hashed)


def _call_timeloop(
    jinja_parse_data: Dict[str, Any],
    rundir: str,
    force_rerun: bool = False,
    n_tries: int = 3,
) -> paretos.Pareto:

    base_dir = os.path.abspath(os.path.dirname(__file__))
    arch_dir = os.path.join(base_dir, "arch.yaml")
    csv_path = os.path.join(rundir, "timeloop-mapper.oaves.csv")
    pkl_path = os.path.join(rundir, "timeloop-mapper.oaves.pkl")
    log_path = os.path.join(rundir, "timeloop-mapper.log")

    # Why use a pickle instead of just reloading the CSV if it's there?
    # The pickle is atomic... if we're interrupted, we'll get an invalid pickle.
    # If Timeloop is interuppted partway through the CSV, we'll get an incomplete CSV.
    if not force_rerun:
        try:
            return pickle.load(open(pkl_path, "rb"))
        except (FileNotFoundError, EOFError):
            pass

    # print(f"Generating {csv_path} with data: {jinja_parse_data}")
    spec = tl.Specification.from_yaml_files(arch_dir, jinja_parse_data=jinja_parse_data)

    success = False
    exc = []

    # I don't know why, but sometimes Timeloop was completing successfully and returning
    # a nonzero exit code, which would propagate an exception here. It happens rarely and
    # nondeterministically, so just rerun Timeloop a couple times if we get that error.
    while not success and n_tries > 0:
        try:
            # print(f'Calling Timeloop in {rundir}')
            tl.call_mapper(spec, rundir, log_to=log_path)
            success = True
        except Exception as e:
            exc.append(e)
            n_tries -= 1

    if not success:
        raise RuntimeError(f"Failed to run Timeloop in {base_dir}") from exc[-1]

    # Commas in imperfect factorization mapping report confuse the CSV format, so
    # change to semicolons
    contents = open(csv_path).read()
    contents = re.sub(r"([A-Z])(\d+),(\d+)", r"\1\2;\3", contents)
    open(csv_path, "w").write(contents)

    df = pd.read_csv(csv_path, header=None)
    pickle.dump(df, open(pkl_path, "wb"))
    # print(f'Finished calling Timeloop in {rundir}')
    return df


def _postprocess_oaves(
    df: pd.DataFrame,
    rank2size: Dict[str, int],
    tensor2ranknames: Dict[str, List[str]],
    tensors_fused: List[str],
    timeloop2name: Dict[str, str],
    uneven_mapping: bool,
    tensors_not_fused: List[str],
    uneven_tensors: List[str],
    rundir: str,
    flipped_tc: bool,
    op_name: str,
) -> pd.DataFrame:

    # Assign headings to the parsed dataframe
    headings = ["Operational Intensity", paretos.UTIL_COL, paretos.ACCESSES_COL]
    for t in tensor2ranknames:
        headings += [f"{t} {paretos.UTIL_COL}"]
        headings += [f"{t} {paretos.ACCESSES_COL}"]

    headings += [paretos.MAPPING_COL, None]
    if len(headings) != len(df.columns):
        raise ValueError(
            f"Can't figure out headings for output in {rundir}. I have {len(headings)}"
            f"heading but the output CSV has {len(df.columns)} columns."
        )
    df.columns = headings
    df = df[df.columns[1:-1]].copy()

    # Calculate the fused mappings
    n_storages_above_fused_loops = 2 + (len(uneven_tensors))
    # Grabs a letter a number if the # is not 1
    loop_regex = re.compile(r"([a-zA-Z])(?!1 )(\d+)")

    def get_fused_loops(tl_mapping: str) -> Tuple[Rank, ...]:
        target_level = tl_mapping.split("L")[n_storages_above_fused_loops]
        loops = tuple(
            Rank(name=timeloop2name[n], size=int(s))
            for n, s in re.findall(loop_regex, target_level)
        )
        return loops

    df[paretos.FUSION_COL] = df[paretos.MAPPING_COL].apply(lambda x: get_fused_loops(x))
    # BUGFIX: MAKE SURE THAT SIZE IS INCLUDED FOR ALL BUT THE
    # LAST (INNERMOST) FUSED LOOP
    df[paretos.FUSION_STR_COL] = df[paretos.FUSION_COL].apply(
        lambda x: " ".join(y.name for y in x)
    )

    tensor_names = sorted(tensors_fused)
    for t in sorted(tensor2ranknames.keys()):
        if t not in tensor_names:
            tensor_names.append(t)

    # Make the mapping output pretty
    tsizes = {
        t: max(math.prod(rank2size[r] for r in ranks), 1)
        for t, ranks in tensor2ranknames.items()
    }
    delete_from_mapping = [
        r" \w1\b(?!\[)",  # Null loops
        r" - L\d+\[\]",  # Empty storages
    ]
    mapping_delete_re = re.compile("(" + r"|".join(delete_from_mapping) + ")")

    keep_columns = list(headings[1:-1]) + [paretos.FUSION_COL, paretos.FUSION_STR_COL]

    df = paretos.Pareto.trim_optimal_df(
        df[keep_columns], flatten_different_fusions=False
    )

    def get_tiling(row):
        (acc, util, tensors) = ([], [], [])
        for t in tensor_names:
            a = row[f"{t} {paretos.ACCESSES_COL}"]
            u = row[f"{t} {paretos.UTIL_COL}"]
            tsize = tsizes[t]
            acc.append(f"{a:.1e}(x{a // tsize})" if a > 0 else "0")
            util.append(f"{u:.1e}(/{tsize // u if u else 0})")
            if t in tensors_fused:
                tensors.append(f"F({t} {tsize:.1e})")
            else:
                tensors.append(f"({t} {tsize:.1e})")

        spec = " ".join(tensors) + f"| DRAM <- {' '.join(acc)} -> L2[{' '.join(util)})]"
        return pretty_sci_notation(spec)

    df[paretos.TILING_COL] = df.apply(lambda row: get_tiling(row), axis=1)
    df[paretos.MAPPING_COL] = df[paretos.MAPPING_COL].apply(
        lambda x: re.sub(mapping_delete_re, "", x)
    )

    df[paretos.FLIPPED_TC_COL] = flipped_tc
    df[paretos.FUSED_TENSOR_COL] = [sorted(tensors_fused)] * len(df)
    for p in paretos.ALL_COLS:
        df[f"{op_name} {p}"] = df[p]

    return df.copy()


def run_timeloop_once(
    rank2size: Dict[str, int],
    tensor2ranknames: Dict[str, List[str]],
    tensors_fused: List[str],
    output_tensors: List[str],
    tensor_core_dims: List[Tuple[List[str], int]],
    chip_datawidth_mult: Dict[str, int],
    bit_ranks: List[str],
    flipped_tc: bool,
    op_name: str,
    uneven_tensors: List[str],
    force_rerun: bool = False,
    uneven_mapping: bool = True,
    n_tries: int = 3,
    return_timeloop_call: bool = False,
) -> Tuple[str, pd.DataFrame]:
    tensors_not_fused = sorted(
        set(t for t in tensor2ranknames if t not in tensors_fused)
    )

    fusable_ranks, unfusable_ranks = set(), set()
    for r in rank2size:
        if (
            tensors_fused
            and all(r in tensor2ranknames[t] for t in tensors_fused)
            and r not in bit_ranks
        ):
            fusable_ranks.add(r)
        else:
            unfusable_ranks.add(r)

    # To ensure our columns line up
    # print(f'Sorting')
    tensor2ranknames = recursive_sort(tensor2ranknames)

    # print(f'Getting translation')
    name2timeloop = get_timeloop_translation(
        tensor2ranknames=tensor2ranknames, output_tensors=output_tensors
    )

    fully_shared_ranks = (
        set.intersection(*[set(s) for s in tensor2ranknames.values()]) - unfusable_ranks
    )

    # print('Translating')
    jinja_parse_data = recursive_translate(
        dict(
            rank2size=rank2size,
            tensor2ranknames=tensor2ranknames,
            fusable_ranks=fusable_ranks,
            unfusable_ranks=unfusable_ranks,
            tensors_fused=tensors_fused,
            tensors_not_fused=tensors_not_fused,
            output_tensors=output_tensors,
            tensor_core_dims=tensor_core_dims,
            chip_datawidth_mult=chip_datawidth_mult,
            uneven_mapping=uneven_mapping,
            bit_ranks=bit_ranks,
            uneven_tensors=sorted(uneven_tensors),
            fully_shared_ranks=sorted(fully_shared_ranks),
        ),
        translation=name2timeloop,
    )

    rundir = get_timeloop_run_dir(jinja_parse_data)

    exc = []

    while n_tries > 0:
        try:
            # print(f'Calling Timeloop in {rundir}')
            n_tries -= 1
            df = _call_timeloop(
                jinja_parse_data=jinja_parse_data,
                rundir=rundir,
                force_rerun=force_rerun,
            )
            time.sleep(1)
            # print(f'Postprocssing oaves')
            return (
                _postprocess_oaves(
                    df=df,
                    rank2size=rank2size,
                    tensor2ranknames=tensor2ranknames,
                    tensors_fused=tensors_fused,
                    timeloop2name={v: k for k, v in name2timeloop.items()},
                    uneven_mapping=uneven_mapping,
                    tensors_not_fused=tensors_not_fused,
                    rundir=rundir,
                    flipped_tc=flipped_tc,
                    op_name=op_name,
                    uneven_tensors=uneven_tensors,
                ),
                rundir,
            )
        except Exception as e:
            exc.append(e)
            print(f"Exception {e}")

    raise RuntimeError(f"Failed to run Timeloop in {rundir}") from exc[-1]


def generate_timeloop_results_for_op(
    op: "Operation",
    uneven_mapping: bool = True,
    flipped_tc: Tuple[bool] = (False, True),
    force_rerun: bool = False,
    must_fuse_tensors: Set[str] = (),
    must_not_fuse_tensors: Set[str] = (),
) -> Union[List[FusedMapping], List[Callable]]:
    # Set up the ranks used in the operation Einsum
    rank2size = defaultdict(lambda: 1)
    for l in op.shared_ranks:
        # Grab the minimum bound. If a shared rank shrinks between two tensors, data
        # is either dropped by this Einsum or added after this Einsum. Either way,
        # we don't need it here.
        input_bounds, output_bounds = l.get_bounds()
        if not input_bounds and not output_bounds:
            raise ValueError(f"Shared rank {l} has no bounds in operation {op}.")
        rank2size[l.name] *= min(
            list(input_bounds.values()) + list(output_bounds.values())
        )
    rank2size = dict(rank2size)

    if len(op.input_tensors) <= 1:
        flipped_tc = [False]

    tensor2ranknames = {
        t.name: [l.name for l in op.shared_ranks if l.overlaps(t)] for t in op.tensors
    }

    # Assign the precision(s) of each tensor
    chip_datawidth_mult = {t.name: 1 for t in op.tensors}
    # If something is getting reduced, we need another precision for accumulation
    if op.get_reduced_ranks():
        for t in op.output_tensors:
            assert t.accum_precision, f"Tensor {t} has unknown accumulation precision."
            assert (
                t.accum_precision % t.precision == 0
            ), f"Tensor {t} accumulation precision {t.accum_precision} not divisible by precision {t.precision}"
            chip_datawidth_mult[t.name] = t.accum_precision // t.precision

    bit_ranks = [f"{t.name} bits" for t in op.tensors]
    for t, b in zip(op.tensors, bit_ranks):
        tensor2ranknames[t.name].append(b)
        rank2size[b] = t.precision

    # Wrapper to parallelize the inner loop
    def _run_timeloop_wrapper(fused_tensor_names, uneven_tensor_names, ftc):
        starttime = time.time()
        tensor_core_dims = get_tensor_core_dims(
            op, tensor2ranknames=tensor2ranknames, flipped_tc=ftc
        )
        try:
            df, rundir = run_timeloop_once(
                rank2size=rank2size,
                tensor2ranknames=tensor2ranknames,
                tensors_fused=list(fused_tensor_names),
                uneven_tensors=list(uneven_tensor_names),
                output_tensors=[t.name for t in op.output_tensors],
                tensor_core_dims=tensor_core_dims,
                chip_datawidth_mult=chip_datawidth_mult,
                force_rerun=force_rerun,
                bit_ranks=bit_ranks,
                uneven_mapping=uneven_mapping,
                flipped_tc=ftc,
                op_name=op.name,
            )
        except Exception as e:
            raise RuntimeError(
                f"Operation {op} failed with fused tensors {fused_tensor_names} "
                f"uneven tensors {uneven_tensor_names} flipped tensor core {ftc}"
            ) from e
        fused_tensors = [t for t in op.tensors if t.name in fused_tensor_names]
        r = []
        if op.force_zero_accesses:
            df[paretos.ACCESSES_COL] = 0
            df[paretos.UTIL_COL] = 0
            df[f"{op.name} {paretos.ACCESSES_COL}"] = 0
            df[f"{op.name} {paretos.UTIL_COL}"] = 0

        df[paretos.ACCESSES_COL] *= op.scale_accesses_by
        df[f"{op.name} {paretos.ACCESSES_COL}"] *= op.scale_accesses_by

        for fs, d in df.groupby(paretos.FUSION_STR_COL, sort=False):
            fused_loops = []
            for f in d.iloc[0][paretos.FUSION_COL]:
                for shared_rank in op.shared_ranks:
                    if f.name == shared_rank.name:
                        fused_loops.append((shared_rank, f))
                        break
                else:
                    raise ValueError(
                        f"Rank {f} has no matching shared_rank in op {op}. Rundir {rundir}."
                        f"Fusion str {d.iloc[0][paretos.FUSION_STR_COL]}. Fusion "
                        f"{d.iloc[0][paretos.FUSION_COL]}"
                    )

            f = FusedMapping(
                fused_tensors=fused_tensors,
                fused_loops=fused_loops,
                op=op,
                mapping_result=Pareto({op.name}, d, flatten_different_fusions=False),
                fusion_string=fs,
            )
            r.append(f)
        # print(f'Op {op.name}: Generated {sum(len(x.df) for x in r)} mappings with {len(r)} fused mappings. {time.time() - starttime:2f} seconds.')
        return r

    fusable_tensors = set(t.name for t in op.tensors) - must_not_fuse_tensors
    tensor_names = set(t.name for t in op.tensors)
    must_fuse_tensors = set(fusable_tensors) & must_fuse_tensors
    mapfuncs = [
        delayed(_run_timeloop_wrapper)(f, un, ftc)
        for ftc in flipped_tc
        for f in powerset(fusable_tensors)
        for un in powerset(tensor_names - set(f))
        if not (must_fuse_tensors - set(f))
    ]
    return mapfuncs


def postprocess_timeloop_results_for_op(
    mappings: List[FusedMapping],
) -> List[FusedMapping]:
    # If there are multiple results with the same fused mapping, combine them here.
    # This can happen if we're exploring flipped_tc or different uneven tensors
    result = defaultdict(list)
    for mapping_group in mappings:
        for m in mapping_group:
            result[str(m)].append(m)

    for v in result.values():
        v[0].mapping_result = Pareto.concat_same(
            [m.mapping_result for m in v],
            flatten_different_fusions=False,
            delay=True,
        )
    return list(v[0] for v in result.values())


def generate_timeloop_results_for_ops(
    ops: List["Operation"],
    uneven_mapping: bool = True,
    flipped_tc: Tuple[bool] = (False, True),
    force_rerun: bool = False,
    must_fuse_easily_fusable: bool = True,
) -> List[List[FusedMapping]]:

    must_fuse_tensors = set()
    must_not_fuse_tensors = set(t.name for t in ops.tensors if not t.is_fusable)
    if must_fuse_easily_fusable:
        # Easily fusable IF:
        # 1. There are 2 or fewer tensors (elementwise op)
        # 2. OR there are >2 tensors and the smallest
        #    is >128x smaller than the next smallest (common in bias ops)
        for op in ops:
            in_tensors = set(t.name for t in op.input_tensors)
            out_tensors = set(t.name for t in op.output_tensors)
            tensor_sizes = sorted(t.size() for t in op.tensors)
            if len(op.input_tensors) <= 1:
                must_fuse_tensors |= in_tensors
            elif tensor_sizes[0] * 128 < tensor_sizes[1]:
                must_fuse_tensors |= in_tensors

    must_fuse_tensors -= must_not_fuse_tensors

    step1 = [
        generate_timeloop_results_for_op(
            op,
            uneven_mapping,
            flipped_tc,
            force_rerun,
            must_fuse_tensors=must_fuse_tensors,
            must_not_fuse_tensors=must_not_fuse_tensors,
        )
        for op in ops
    ]

    flattened = [m for op in step1 for m in op]
    called = parallel_proc(flattened, pbar="Calling Timeloop")

    result = []
    for op in step1:
        result.append(called[: len(op)])
        called = called[len(op) :]

    assert len(step1) == len(result)
    for s, r in zip(step1, result):
        assert len(s) == len(r)
    result = [postprocess_timeloop_results_for_op(r) for r in result]
    flattened = [m.mapping_result for r in result for m in r]
    called = parallel_proc(flattened, pbar="Postprocessing solutions")

    i = 0
    for r in result:
        for r2 in r:
            r2.mapping_result = called[i]
            i += 1
    assert i == len(called)

    return result
