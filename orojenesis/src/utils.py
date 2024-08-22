#!/usr/bin/env python3

import os
import platform
import re
import subprocess
import pathlib
import pandas as pd
import numpy as np
import yaml
import logging
import sys
import shutil
import time
from dataclasses import dataclass
from abc import ABC, abstractmethod

from . import orojenesis_process_data

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s - (%(filename)s:%(lineno)d)"
)

try:
    _ = os.path.expanduser(os.environ["TIMELOOP_BASE_PATH"])
except KeyError:
    logger.error("Please specify TIMELOOP_BASE_PATH!")


def store_yaml(yaml_path, data):
    with open(yaml_path, "w") as f:
        yaml.dump(data, f)


def parse_yaml(yaml_path):
    with open(yaml_path, "r") as f:
        data = yaml.full_load(f)
        return data


class Op(ABC):
    "Template for defining Einsum."

    @abstractmethod
    def to_str(self):
        pass

    @abstractmethod
    def to_dict(self):
        pass


    @abstractmethod
    def get_tensor_size(self):
        pass

    @abstractmethod
    def get_compute_size(self):
        pass

    def to_yaml(self, yaml_path):
        prob = self.to_dict()
        logger.info(f"GenProblemYAML> {yaml_path}")
        store_yaml(yaml_path, prob)

    def get_op_int(self):
        return self.get_compute_size() * 2 / self.get_tensor_size()[0]



@dataclass
class Conv(Op):
    "Timeloop Conv problem definition"
    R: int = 1
    S: int = 1
    P: int = 1
    Q: int = 1
    C: int = 1
    K: int = 1
    N: int = 1
    Wstride: int = 1
    Hstride: int = 1
    Wdilation: int = 1
    Hdilation: int = 1

    def to_str(self):
        return f"{self.R}_{self.S}_{self.P}_{self.Q}_{self.C}_{self.K}_{self.N}_{self.Wstride}_{self.Hstride}_{self.Wdilation}_{self.Hdilation}"

    def to_dict(self):
        d = {"problem": self.__dict__}
        d["problem"]["shape"] = "cnn-layer"
        return d

    def get_tensor_size(self):
        W = self.R * self.S * self.C * self.K
        P_in = (self.P - 1) * self.Wstride + self.R
        Q_in = (self.Q - 1) * self.Hstride + self.S
        I = P_in * Q_in * self.C * self.N
        O = self.P * self.Q * self.K * self.N
        return W + I + O, W, I, O

    def get_compute_size(self):
        return self.R * self.S * self.P * self.Q * self.C * self.K * self.N


@dataclass
class GBMM(Op):
    "Timeloop Grouped Batch Matrix Multiplication problem definition. Note that the compute reported for grouped Einsum is incorrect."
    M: int = 1
    K: int = 1
    N: int = 1
    H: int = 1
    G: int = 1

    def to_str(self):
        return f"gbmm_{self.M}_{self.K}_{self.N}_{self.H}_{self.G}"

    def to_dict(self):
        template = "./configs/gbmm_template.yaml"
        d = parse_yaml(template)
        d["problem"]["instance"] = self.__dict__
        return d

    def get_tensor_size(self):
        W = self.K * self.N * self.H // self.G
        I = self.M * self.K * self.H
        O = self.M * self.N * self.H
        return W + I + O, W, I, O

    def get_compute_size(self):
        return self.M * self.K * self.N * self.H


def GenProblemYAML(prob_dims, yaml_path):
    prob = Conv(**prob_dims).to_dict()
    logger.info(f"GenProblemYAML> {yaml_path}")
    store_yaml(yaml_path, prob)


def RunMapper(
    arch_yaml,
    workload_yaml,
    mapper_yaml,
    output_dir,
    cwd=os.getcwd(),
    stdout=None,
    stderr=None,
):
    try:
        timeloop = os.environ["TIMELOOP_BASE_PATH"]
        cmd = f"{timeloop}/bin/timeloop-mapper {arch_yaml} {workload_yaml} {mapper_yaml} -o {output_dir}"
        logger.info(f"RunMapper> {cmd}")
        p = subprocess.check_call(cmd.split(" "), cwd=cwd, stdout=stdout, stderr=stderr)
    except:
        logger.error("RunMapper Failed!")
        raise Exception("RunMapper Failed!")


def RunOrojenesisPostProcessing(
    output_dir,
    cwd=os.getcwd(),
    stdout=None,
    stderr=None,
    keep_one_best_entry=True,
    keep_all_entry=False,
    keep_one_best_entry_across_buf=False,
):

    try:
        preprocessed_csv = output_dir / "timeloop-mapper.orojenesis.csv"
        orojenesis_csv = output_dir / "orojenesis.csv"
        logger.info(
            f"RunOrojenesisPostProcessing> process {preprocessed_csv} to {orojenesis_csv}"
        )
        orojenesis_process_data.process_data(
            preprocessed_csv,
            orojenesis_csv,
            keep_one_best_entry=keep_one_best_entry,
            keep_all_entry=keep_all_entry,
            keep_one_best_entry_across_buf=keep_one_best_entry_across_buf,
        )
    except:
        raise Exception("RunOrojenesisPostProcessing Failed!")


def GenerateBound(
    prob,
    output_dir,
    arch_yaml="./configs/single-einsum/arch.yaml",
    mapper_yaml="./configs/single-einsum/conv_mapper.yaml",
    force_rerun=False,
    force_rerun_postprocess=False,
    keep_one_best_entry=True,
    keep_all_entry=False,
    keep_one_best_entry_across_buf=False,
):
    output_dir_se = output_dir / prob.to_str()
    output_dir_se.mkdir(parents=True, exist_ok=True)
    workload_yaml = output_dir_se / "problem.yaml"
    prob.to_yaml(workload_yaml)
    preprocessed_csv = output_dir_se / "timeloop-mapper.orojenesis.csv"

    start_time = time.time()
    if force_rerun or not preprocessed_csv.exists():
        RunMapper(arch_yaml, workload_yaml, mapper_yaml, output_dir_se)

    postprocessed_csv = output_dir_se / "orojenesis.csv"
    if force_rerun or force_rerun_postprocess or not postprocessed_csv.exists():
        RunOrojenesisPostProcessing(
            output_dir_se,
            keep_one_best_entry=keep_one_best_entry,
            keep_all_entry=keep_all_entry,
            keep_one_best_entry_across_buf=keep_one_best_entry_across_buf,
        )
    end_time = time.time()
    dur = end_time - start_time
    logger.info(f"GenerateBounds> Duration: {dur:.2f}s.")


def process_dataframe(df, scale=1, get_opt=True, get_mapping=False):
    df_T = df.T
    data = df_T.values.tolist()

    d = {"Op_Intensity": data[0], "DRAM_Accesses": data[2]}
    d["mapping"] = data[-2]
    df = pd.DataFrame(d, index=data[1]).sort_index()
    df_max_op_traf = df.copy()
    max_op = 0
    max_idx = 0
    min_traf = []
    max_op_int = []
    max_index = []
    mappings = []
    best_data = None
    if get_opt:
        for i, row in df_max_op_traf.iterrows():
            cur_val = row["Op_Intensity"]
            if cur_val > max_op:
                best_data = row
                max_op = cur_val
                max_index.append(i)
                max_op_int.append(row["Op_Intensity"] / float(scale))
                min_traf.append(row["DRAM_Accesses"] * float(scale))
                mappings.append(row["mapping"])
            df_max_op_traf.at[i, df.columns[0]] = max_op_int[-1]
            df_max_op_traf.at[i, df.columns[1]] = min_traf[-1]
            df_max_op_traf.at[i, df.columns[2]] = mappings[-1]

        spatial_factor = 1
        d = {"Op_Intensity": max_op_int, "DRAM_Accesses": min_traf, "mapping": mappings}
        df_max_only_traf = pd.DataFrame(d, index=max_index)
        return df_max_only_traf
    else:
        return df_max_op_traf


def process_data(stats_file, scale=1, get_opt=True, get_mapping=False):
    stats_file = str(stats_file)
    if stats_file.endswith(".csv"):
        df = pd.read_csv(stats_file, header=None)
        return process_dataframe(df, scale=scale, get_opt=get_opt, get_mapping=get_mapping)
    else:
        raise Exception("Invalid stats file input!")


def get_stats_files(output_dir, probs, pareto_optimal=True):
    if pareto_optimal:
        return [output_dir / prob.to_str() / "orojenesis.csv" for prob in probs]
    else:
        return [output_dir / prob.to_str() / "timeloop-mapper.orojenesis.csv" for prob in probs]


def get_dfs(stats_files, scales=None, get_opt=True, get_mapping=False):
    dfs = []
    for i, stats_file in enumerate(stats_files):
        df = process_data(stats_file, get_opt=get_opt, get_mapping=get_mapping)
        dfs.append(df)
    return dfs


def get_prob_M_dim(pair_chains, graph_dir, layers_dict):
    P_prob = None
    for chain_idx, pair_chain in enumerate(pair_chains):
        for pair_idx, pair in enumerate(pair_chain):
            for prob_idx, prob in enumerate(pair):
                if "mm" in prob:
                    prob_path = graph_dir / (layers_dict[prob] + ".yaml")
                    P_prob = int(parse_yaml(prob_path)["problem"]["instance"]["M"])
                    break
    return P_prob


def check_chain_valid(subchain, layers_dict, graph_dir):
    # Process the min accesses for each subchain
    for pair_idx, pair in enumerate(subchain):
        prev_dims = []
        for prob_idx, prob in enumerate(pair):
            if "prob_" in layers_dict[prob]:
                prob_shape = parse_yaml(graph_dir / (layers_dict[prob]+'.yaml'))
                dims = prob_shape['problem']['instance']
                if 'H' in dims.keys(): # for BMM
                    input_size = dims['H'] * dims['M'] * dims['K']
                    output_size = dims['H'] * dims['M'] * dims['N']
                else: # for MM
                    input_size =  dims['M'] * dims['K']
                    output_size = dims['M'] * dims['N']

                if prob_idx != 0:
                    if input_size != prev_output_size:
                        return False
                prev_output_size = output_size
    return True


def get_valid_chains(chains, layers_dict, graph_dir):
    chain_idx_tuple = [(chain_idx, chain) for chain_idx, chain in enumerate(chains)]
    nonempty_chains = [lst for lst in chain_idx_tuple if lst[1] != []]
    chains = nonempty_chains

    valid_chains = []
    for chain in nonempty_chains:
        if check_chain_valid(chain[1], layers_dict, graph_dir):
            valid_chains.append(chain)
    return valid_chains


def compute_num_head_factors(prob, num_heads, buffer_all, batch):
    input_factor = 1
    output_factor = 1
    weight_access_factor = 1
    weight_buf_factor = 1
    if prob in ["mm_proj_0", "mm_proj_1", "mm_proj_2", "mm_qk", "mm_qkv"]:
        output_factor = num_heads
    if prob in ["mm_qk", "mm_qkv", "mm_proj_final"]:
        input_factor = num_heads
    if prob in [
        "mm_proj_0",
        "mm_proj_1",
        "mm_proj_2",
        "mm_qk",
        "mm_qkv",
        "mm_proj_final",
    ]:
        # assuming multiple heads increases both buffer req and dram accesses
        weight_access_factor = num_heads
        # NOTE keep the postprocessing in porcess tiled fusion script for now
        # if prob in ["mm_qk", "mm_qkv"]:
        #     weight_access_factor *= 1 #  batch the reason why we don't multiply it is because batch of 16 is already taken into account when P at L2 is larger than 16
        #     if buffer_all:
        #         weight_access_factor *= batch # if weight is fully buffered, we need to access them batch times
        if buffer_all: # or prob in ["mm_qk", "mm_qkv"]:
            weight_buf_factor = num_heads
        else:
            weight_buf_factor = 1
    return input_factor, output_factor, weight_access_factor, weight_buf_factor


def df_one_best_entry_across_buf(df, sort_metric, metric):
    df = df.sort_values(by=[sort_metric])
    min_accesses_val = None
    min_indices = []
    for i, row in df.iterrows():
        cur_val = row[metric]
        if min_accesses_val is None:
            min_accesses_val = cur_val
            min_indices.append(i)
        else:
            if cur_val < min_accesses_val:
                min_accesses_val = cur_val
                min_indices.append(i)
    df = df.loc[min_indices]
    return df


def row_to_result_dict(
    matched_row, total_levels=3, weight_level=0, input_level=1, output_level=1
):
    buf_size = matched_row[0]
    total_accesses = matched_row[2]
    mapping = matched_row[3]
    weight_util_idx = 6 + weight_level * 3 + 0
    weight_util = matched_row[weight_util_idx]
    input_util_idx = 6 + input_level * 3 + 1
    input_util = matched_row[input_util_idx]
    output_util_idx = 6 + output_level * 3 + 2
    output_util = matched_row[output_util_idx]
    dram_accesses_idx = 6 + total_levels * 3
    weight_accesses = matched_row[dram_accesses_idx + 0]
    input_accesses = matched_row[dram_accesses_idx + 1]
    output_accesses = matched_row[dram_accesses_idx + 2]
    result = {
        "buf_size": buf_size,
        "total_accesses": total_accesses,
        "weight_accesses": weight_accesses,
        "input_accesses": input_accesses,
        "output_accesses": output_accesses,
        "input_util": input_util,
        "weight_util": weight_util,
        "output_util": output_util,
        "mapping": mapping,
    }
    return result


def map_index_to_tensor_names(
    total_levels=3, weight_level=0, input_level=1, output_level=1
):
    dram_accesses_idx = 6 + total_levels * 3
    index_to_tensor_names = {
        0: "buf_size",
        1: "OI",
        2: "total_accesses",
        3: "mapping",
        5: "compute",
        dram_accesses_idx + 0: "weight_accesses",
        dram_accesses_idx + 1: "input_accesses",
        dram_accesses_idx + 2: "output_accesses",
        6 + input_level * 3 + 1: "input_util",
        6 + weight_level * 3 + 0: "weight_util",
        6 + output_level * 3 + 2: "output_util",

    }

    tensor_names_to_index = {value: key for key, value in index_to_tensor_names.items()}
    return index_to_tensor_names, tensor_names_to_index


def update_df_metric(df, tensor_names_to_index, metric_name, factor):
    metric_idx = tensor_names_to_index[metric_name]
    df[metric_idx] = df[metric_idx] * factor


def update_df_metric_sum(df, tensor_names_to_index, target_metric_name, sum_metric_names):
    total_val = 0
    for sum_metric in sum_metric_names:
        sum_metric_idx = tensor_names_to_index[sum_metric]
        total_val += df[sum_metric_idx]

    target_metric_idx = tensor_names_to_index[target_metric_name]
    df[target_metric_idx] = total_val


def merge_df_slices(dfs):
    df = pd.concat(dfs)
    df = df.reset_index(drop=True)
    # idx = df.groupby('max_buf_size')['fused_accesses'].idxmin()
    # df = df.loc[idx]
    # df = df.sort_values(by=['max_buf_size'])
    df = df_one_best_entry_across_buf(df, "max_buf_size", "fused_accesses")
    return df


def keep_best_among_group(
    df, group_column="max_buf_size", value_column="fused_accesses"
):
    df[group_column] = df[group_column].astype('int64')
    df[value_column] = df[value_column].astype('int64')

    idx = df.groupby(group_column)[value_column].idxmin()
    df = df.loc[idx]
    return df


def merge_df(df_total_fused, df, m_value):
    logger.debug(f"BEFORE\n{df_total_fused}")
    logger.debug(f"MERGE\n{df}")

    df_total_fused = keep_best_among_group(df_total_fused)
    df = keep_best_among_group(df)

    df_total_fused_new = pd.DataFrame()
    i = 0
    j = 0
    df_0_size = df_total_fused.shape[0]
    df_1_size = df.shape[0]
    bufsize0 = df_total_fused.iloc[i]["max_buf_size"]
    bufsize1 = df.iloc[j, 0]

    # assert(bufsize0 == bufsize1)

    while True:
        new_row = None
        if i >= df_0_size and j >= df_1_size:
            break
        elif i < df_0_size and j >= df_1_size:
            bufsize0 = df_total_fused.iloc[i]["max_buf_size"]
            new_row = df_total_fused.iloc[i] + df.iloc[df_1_size - 1]
            new_row["max_buf_size"] = bufsize0
            i += 1

        elif i >= df_0_size and j < df_1_size:
            bufsize1 = df.iloc[j, 0]
            new_row = df_total_fused.iloc[df_0_size - 1] + df.iloc[j]
            new_row["max_buf_size"] = bufsize1
            j += 1

        elif i < df_0_size and j < df_1_size:
            # bufsize0 = df_total_fused.iloc[i,0]
            bufsize0 = df_total_fused.iloc[i]["max_buf_size"]
            bufsize1 = df.iloc[j]["max_buf_size"]
            # print(bufsize0, bufsize1)
            if bufsize1 == bufsize0:
                # print(df.iloc[[j]]['orig_accesses'])
                # orig_accesses = df_total_fused.iloc[i, orig_accesses_idx] + df.iloc[j, orig_accesses_idx]
                # new_row =  pd.DataFrame([{'bufsize': bufsize0, 'orig_accesses': orig_accesses}])
                new_row = df_total_fused.iloc[i] + df.iloc[j]
                new_row["max_buf_size"] = bufsize0
                # if i + 1 < df_0_size:
                i += 1
                # if j + 1 < df_1_size:
                j += 1
            elif bufsize1 > bufsize0:
                if j > 0:
                    new_row = df_total_fused.iloc[i] + df.iloc[j - 1]
                    new_row["max_buf_size"] = bufsize0
                # orig_accesses = df_total_fused.iloc[i, orig_accesses_idx] + df.iloc[j-1, orig_accesses_idx]
                # new_row =  pd.DataFrame([{'bufsize': bufsize0, 'orig_accesses': orig_accesses}])
                # if i + 1 < df_0_size:
                i += 1
            elif bufsize1 < bufsize0:
                if i > 0:
                    new_row = df_total_fused.iloc[i - 1] + df.iloc[j]
                    new_row["max_buf_size"] = bufsize1
                # orig_accesses = df_total_fused.iloc[i-1, orig_accesses_idx] + df.iloc[j, orig_accesses_idx]
                # new_row =  pd.DataFrame([{'bufsize': bufsize1, 'orig_accesses': orig_accesses}])
                # if j + 1 < df_1_size:
                j += 1
        # df_total_fused_new = pd.concat([df_total_fused_new, new_row.T], axis=0).reset_index(drop=True)
        if new_row is not None:
            new_row["M1"] = m_value
            # df_total_fused_new = df_total_fused_new.append(new_row).
            df_total_fused_new = pd.concat(
                [df_total_fused_new, new_row.to_frame().T], ignore_index=True
            )

    df_total_fused = df_total_fused_new
    df_total_fused = df_total_fused.sort_values(by=["max_buf_size"])

    logger.debug(f"AFTER\n {df_total_fused}")
    return df_total_fused


def gen_prob_chain(graph_dir, input_format=None):
    if input_format is None:
        chains = parse_yaml(graph_dir / "chains.yaml")
    else:
        chains = parse_yaml(graph_dir / f"{input_format}.yaml")
    layers_dict = parse_yaml(graph_dir / "layers_dict.yaml")
    chains = get_valid_chains(chains, layers_dict, graph_dir)

    prob_chains = []
    for chain in chains:
        prob_chain = []
        for sub_chain in chain[1]:
            sub_prob_chain = []
            for item in sub_chain:
                sub_prob_chain.append({item: layers_dict[item]})
            prob_chain.append(sub_prob_chain)
        prob_chains.append(prob_chain)
    return chains, layers_dict, prob_chains


def compute_reduction(df_baseline, df):
    logger.debug(f"BEFORE\n{df_baseline}")
    logger.debug(f"MERGE\n{df}")
    df_red = pd.DataFrame()
    i = 0
    j = 0
    df_0_size = df_baseline.shape[0]
    df_1_size = df.shape[0]
    bufsize0 = df_baseline.iloc[i]["max_buf_size"]
    bufsize1 = df.iloc[j, 0]

    while True:
        new_row = None
        max_buf_size = None
        if i >= df_0_size and j >= df_1_size:
            break
        elif i < df_0_size and j >= df_1_size:
            max_buf_size = df_baseline.iloc[i]["max_buf_size"]
            df_baseline_idx = i
            df_idx = df_1_size - 1
            i += 1
        elif i >= df_0_size and j < df_1_size:
            max_buf_size = df.iloc[j, 0]
            df_baseline_idx = df_0_size - 1
            df_idx = j
            j += 1
        elif i < df_0_size and j < df_1_size:
            bufsize0 = df_baseline.iloc[i]["max_buf_size"]
            bufsize1 = df.iloc[j]["max_buf_size"]
            if bufsize1 == bufsize0:
                df_baseline_idx = i
                df_idx = j
                max_buf_size = bufsize0
                i += 1
                j += 1
            elif bufsize1 > bufsize0:
                if j > 0:
                    df_baseline_idx = i
                    df_idx = j - 1
                    max_buf_size = bufsize0
                i += 1
            elif bufsize1 < bufsize0:
                if i > 0:
                    df_baseline_idx = i - 1
                    df_idx = j
                    max_buf_size = bufsize1
                j += 1

        if max_buf_size is not None:
            baseline_accesses = df_baseline.iloc[df_baseline_idx]["fused_accesses"]
            compare_accesses = df.iloc[df_idx]["fused_accesses"]
            red_accesses = baseline_accesses - compare_accesses
            red_ratio = red_accesses / baseline_accesses
            accesses_ratio = baseline_accesses / compare_accesses
            result_dict = {
                "max_buf_size": max_buf_size,
                "baseline_accesses": baseline_accesses,
                "compare_accesses": compare_accesses,
                "red_accesses": red_accesses,
                "red_ratio": red_ratio,
                "accesses_ratio": accesses_ratio,
            }
            new_row = pd.DataFrame([result_dict])
            df_red = pd.concat([df_red, new_row], ignore_index=True)

    df_red = df_red.sort_values(by=["max_buf_size"])

    logger.debug(f"AFTER\n {df_red}")
    return df_red


def args_to_str(
    num_heads=32,
    batch_size=16,
    matmul_only=False,
    ar=False,
    nored="nored",
    spatial_factor=None,
):
    head_str = f"_h{num_heads}"
    batch_str = f"_b{batch_size}"
    matmul_str = "_matmul" if matmul_only else ""
    ar_str = "autoregressive_" if ar else ""
    nored_str = f"_{nored}" if nored != "" else ""
    spatial_str = f"_s{spatial_factor}" if spatial_factor is not None else ""
    return head_str, batch_str, matmul_str, ar_str, nored_str, spatial_str


def get_workload_path(workload_dir, model_name, batch_size, matmul_only, ar, nored):
    _, batch_str, matmul_str, ar_str, nored_str, _ = args_to_str(
        batch_size=batch_size, matmul_only=matmul_only, ar=ar, nored=nored
    )
    graph_dir = pathlib.Path(
        f"{workload_dir}/{model_name}_graph{batch_str}{ar_str}{matmul_str}{nored_str}"
    )
    return graph_dir


def get_chain_config(
    workload_dir,
    output_dir,
    model_name="gpt3-6.7b",
    input_format="chain",
    num_heads=32,
    batch_size=16,
    matmul_only=True,
    ar=False,
    nored="nored",
):
    graph_dir = get_workload_path(
        workload_dir, model_name, batch_size, matmul_only, ar, nored
    )
    input_format_str = "chains" if input_format == "chain" else input_format
    chains, layers_dict, prob_chains = gen_prob_chain(
        graph_dir, input_format=input_format_str
    )
    return chains, layers_dict, prob_chains


def get_output_path(
    chain_idx,
    output_dir,
    model_name="gpt3-6.7b",
    input_format="opt_schedules_mm",
    constraint_config="_relax_io_kn",
    scheme="2",
    num_heads=32,
    batch_size=16,
    matmul_only=True,
    ar=False,
    nored="nored",
    spatial_factor=None,
    eval_slices=False,
    enable_fusion=True,
    arch_prefix="",
):
    fusion_str = "_fused" if enable_fusion else "_unfused"
    slice_str = "_slice" if eval_slices else ""
    head_str, batch_str, matmul_str, ar_str, nored_str, spatial_str = args_to_str(
        num_heads=num_heads, batch_size=batch_size, matmul_only=matmul_only, ar=ar, nored=nored
    )
    output_csv = f"{output_dir}/{arch_prefix}{model_name}/scheme{scheme}{head_str}{batch_str}{ar_str}{matmul_str}{nored_str}{constraint_config}/{input_format}{fusion_str}_{chain_idx}{slice_str}_scheme{scheme}.csv"
    return output_csv


def get_access(df, buf_size, key="DRAM_Accesses"):
    row = df[df.index <= buf_size].iloc[-1]
    return row[key]


def lookup_metric(df, buf_size, key="Op_Intensity"):
    data = []
    for num in buf_size:
        metric = get_access(df, num, key=key)
        data.append(metric)
    return np.array(data)


def derive_performance_bounds(
    df, area_per_mac, area_per_B, freq, bw_Bps, total_area, total_compute
):
    df = df.groupby("max_buf_size")["fused_accesses"].min().reset_index()
    df.set_index("max_buf_size", inplace=True)

    df["Op_Intensity"] = total_compute / df["fused_accesses"]
    df["DRAM_Accesses"] = df["fused_accesses"]

    min_buf = df.index.min()
    max_buf_size = total_area / area_per_B
    buf_size = np.linspace(
        min_buf, max_buf_size, 100
    )  # Adjust the range and number of points as needed
    total_buf_area = buf_size * area_per_B
    total_mac_area = total_area - total_buf_area
    buf_ratio = total_buf_area / total_area
    OI = lookup_metric(df, buf_size, key="Op_Intensity")
    mem_bound_perf = OI * bw_Bps  # OP/s
    num_mac = total_mac_area / area_per_mac
    compute_bound_perf = num_mac * freq
    perf = np.minimum(mem_bound_perf, compute_bound_perf)

    intersection = None
    for i, x in enumerate(buf_size):
        if compute_bound_perf[i] <= mem_bound_perf[i]:
            intersection = x
            break
    return mem_bound_perf, compute_bound_perf, perf, buf_ratio, intersection


def parse_spatial_factors(mapping_str):
    mappings = mapping_str.split(' - ')
    spatial_factor_dict = {}
    spatial_all  = 1

    for text in mappings:
        m = re.search(r'L(.*?)\[', text)
        level = int(m.group(1))
        m = re.search(r'\[(.*?)\]', text)
        spatial_factor_dict[level] = {}
        keep_str = m.group(1)
        if 'W' in keep_str:
            spatial_factor_dict[level]['keep_W'] = True
        if 'I' in keep_str:
            spatial_factor_dict[level]['keep_I'] = True
        if 'O' in keep_str:
            spatial_factor_dict[level]['keep_O'] = True

        matches = re.findall(r' ([A-Z])(\d+)X', text)

        total_spatial = 1

        for m in matches:
            spatial_factor_dict[level][m[0]] = int(m[1])
            total_spatial *= int(m[1])
        spatial_factor_dict[level]['spatial'] = total_spatial
        spatial_all *= total_spatial
    return spatial_factor_dict, spatial_all


def prime_factorize(n):
  """
  This function factors a given number (n) into its prime factors.

  Args:
      n (int): The number to be factored.

  Returns:
      list: A list containing the prime factors of n.
  """

  if n <= 1:
    return []
  factors = []
  i = 2
  while i * i <= n:
    if n % i == 0:
      factors.append(i)
      n //= i
    else:
      i += 1
  if n > 1:
    factors.append(n)
  return factors
