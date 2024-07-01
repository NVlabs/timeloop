#!/usr/bin/env python3
import pathlib
import math
import fire
import numpy as np
import pandas as pd
from collections import OrderedDict
from . import utils

logger = utils.logger


def process_untiled_fusion(
    workload_dir,
    output_dir,
    model_name="gpt3-6.7b",
    input_format="chain",
    num_heads=32,
    batch=16,
    matmul_only=False,
    ar="",
    nored="nored",
    spatial_factor=None,
    full_mapspace=False,
    arch_prefix="",
    data_bytes=2,
):

    nored = f"_{nored}" if nored != "" else ""
    batch = f"_b{batch}" if batch != "" else ""
    matmul = "_matmul" if matmul_only else ""
    scheme = f"opt"
    heads = f"_h{num_heads}"
    spatial = f"_s{spatial_factor}" if spatial_factor is not None else ""
    ar = "autoregressive_" if ar else ""

    # input_format options include ['chain', 'opt_schedules_mm']
    batch_size_postprocessing = 1  # batchsize for post processing
    if batch != "":
        bmm_weight_loads = int(batch.split("b")[-1])
    else:
        bmm_weight_loads = batch_size_postprocessing

    if arch_prefix == "simba_":
        total_levels = 6
        weight_level = 2
        input_level = 4
        output_level = 4
        keep_one_best_entry_across_buf = False
    else:
        total_levels = 3
        weight_level = 0
        input_level = 1
        output_level = 1
        keep_one_best_entry_across_buf = True

    if full_mapspace:
        oaves_file = "timeloop-mapper.oaves.csv"
    else:
        oaves_file = "oaves.csv"

    graph_dir = pathlib.Path(
        f"{workload_dir}/{model_name}_graph{batch}{ar}{matmul}{nored}"
    )
    layers_dict = utils.parse_yaml(graph_dir / "layers_dict.yaml")
    pair_chains = utils.parse_yaml(graph_dir / "pair_chains.yaml")

    if input_format == "chain":
        chains = utils.parse_yaml(graph_dir / "chains.yaml")
    else:
        chains = utils.parse_yaml(graph_dir / f"{input_format}.yaml")

    chain_tuples = utils.get_valid_chains(chains, layers_dict, graph_dir)
    chains = [chain_tuple[1] for chain_tuple in chain_tuples]

    # merge subchains for
    merge_chains = []
    for chain in chains:
        merge_chain = [[]]
        for subchain in chain:
            for item in subchain:
                merge_chain[-1].append(item)
        merge_chains.append(merge_chain)

    cur_dir = pathlib.Path(
        f"{output_dir}/{arch_prefix}{model_name}/scheme{scheme}{heads}{batch}{spatial}{ar}{matmul}{nored}"
    )
    cur_dir.mkdir(exist_ok=True, parents=True)

    output_subdir = pathlib.Path(
        f"{output_dir}/{arch_prefix}{model_name}/oaves_chain_2levels_outputs{batch}{spatial}_opt"
    )

    pd.set_option("display.max_columns", None)  # Display all columns without truncation
    pd.set_option("display.width", None)  # Allow the DataFrame to display in full width
    pd.set_option(
        "display.max_rows", None
    )  # Display all rows without truncation (optional)

    for enable_fusion in [False, True]:
        fusion_str = "_fused" if enable_fusion else "_unfused"
        for chain_idx, subchain in enumerate(merge_chains):
            df_total_fused = None
            logger.info(f"Chain{chain_idx}: {subchain}")

            # Process the min accesses for each subchain
            fused_results = {
                "orig_accesses": 0,
                "fused_accesses": 0,
                "max_buf_size": None,
            }
            chain_dims = []
            for pair_idx, pair in enumerate(subchain):
                optimal_accesses_sub_chain = 0
                optimal_accesses_sub_chain_fused = 0
                logger.info(f"Subchain{pair_idx}: {pair}")
                subchain_dims = []
                for prob_idx, prob in enumerate(pair):
                    dims = layers_dict[prob].replace("prob_", "").split("_")
                    dims = [int(dim) for dim in dims]
                    subchain_dims.append(dims)
                chain_dims.append(subchain_dims)

            # Process each subchain
            for pair_idx, pair in enumerate(subchain):
                logger.info(f"Chain{chain_idx} Subchain{pair_idx}: {pair}")
                assert pair_idx < 1
                for prob_idx, prob in enumerate(pair):
                    oaves_csv = output_subdir / layers_dict[prob] / oaves_file

                    logger.info(f"Parse CSV: {oaves_csv}")
                    df = pd.read_csv(oaves_csv, header=None)
                    df = df.reset_index(drop=True)
                    if matmul_only:
                        (
                            input_factor,
                            output_factor,
                            weight_access_factor,
                            weight_buf_factor,
                        ) = utils.compute_num_head_factors(
                            prob, num_heads, buffer_all=True, batch=bmm_weight_loads
                        )
                    else:
                        input_factor, output_factor, weight_access_factor, weight_buf_factor = 1, 1, 1, 1

                    results = []
                    for _, matched_row in df.iterrows():
                        result = utils.row_to_result_dict(
                            matched_row,
                            total_levels=total_levels,
                            weight_level=weight_level,
                            input_level=input_level,
                            output_level=input_level,
                        )
                        fused_results = {
                            "orig_accesses": 0,
                            "fused_accesses": 0,
                            "max_buf_size": None,
                            "mapping": result["mapping"] + "|",
                        }
                        fused_results["orig_accesses"] = (
                            result["weight_accesses"] * weight_access_factor
                            + result["input_accesses"] * input_factor * batch_size_postprocessing
                            + result["output_accesses"] * output_factor * batch_size_postprocessing
                        )
                        fused_results["fused_accesses"] = fused_results["orig_accesses"]
                        weight_buffer_size = result["weight_util"] * weight_buf_factor
                        input_buffer_size = (
                            result["input_util"] * input_factor * batch_size_postprocessing
                        )
                        output_buffer_size = (
                            result["output_util"] * output_factor * batch_size_postprocessing
                        )
                        buffer_size = (
                            weight_buffer_size + input_buffer_size + output_buffer_size
                        )
                        interlayer_weight_util = [weight_buffer_size]
                        fused_results["interlayer_weight_util"] = interlayer_weight_util
                        fused_results["max_buf_size"] = buffer_size

                        # logger.info(f'RESULT {result}')
                        # logger.info(f'RESULT fused {weight_buffer_size}, {input_buffer_size}, {output_buffer_size}')
                        # logger.info(f'RESULT fused {fused_results}')
                        fused_results["M1"] = ""

                        if not enable_fusion:
                            results.append(fused_results)

                        if enable_fusion:
                            dims = chain_dims[pair_idx][prob_idx]

                            # print(chain_idx, pair_idx, prob_idx, dims, len(pair), input_factor, output_factor)
                            input_tensor_size = (
                                dims[0] * dims[1] * input_factor * batch_size_postprocessing * data_bytes
                            )
                            output_tensor_size = (
                                dims[0] * dims[2] * output_factor * batch_size_postprocessing * data_bytes
                            )
                            weight_tensor_size = dims[1] * dims[2] * weight_buf_factor * data_bytes
                            if prob_idx != 0 and prob_idx < len(pair) - 1:
                                #  print('skip both')
                                fused_results = {
                                    "orig_accesses": 0,
                                    "fused_accesses": 0,
                                    "max_buf_size": None,
                                    "mapping": result["mapping"] + "|",
                                }
                                fused_results["fused_accesses"] = (
                                    result["weight_accesses"] * weight_access_factor
                                )
                                fused_results["orig_accesses"] = fused_results[
                                    "fused_accesses"
                                ]
                                assert (
                                    fused_results["orig_accesses"] >= weight_tensor_size / data_bytes
                                )
                                weight_buffer_size = (
                                    result["weight_util"] * weight_buf_factor
                                )
                                # It requires both input and output buffer size as the mapping is flexible
                                buffer_size = (
                                    weight_buffer_size
                                    + input_tensor_size
                                    + output_tensor_size
                                )
                                fused_results["interlayer_weight_util"] = [
                                    weight_buffer_size
                                ]
                                fused_results["max_buf_size"] = buffer_size
                                fused_results["M1"] = ""
                                results.append(fused_results)

                            if prob_idx != 0:  # first layer cannot skip input accesses
                                # print('skip inputs')
                                fused_results = {
                                    "orig_accesses": 0,
                                    "fused_accesses": 0,
                                    "max_buf_size": None,
                                    "mapping": result["mapping"] + "|",
                                }
                                fused_results["fused_accesses"] = (
                                    result["weight_accesses"] * weight_access_factor
                                    + result["output_accesses"]
                                    * output_factor
                                    * batch_size_postprocessing
                                )
                                fused_results["orig_accesses"] = fused_results[
                                    "fused_accesses"
                                ]
                                assert (
                                    fused_results["orig_accesses"]
                                    >= (weight_tensor_size + output_tensor_size) / data_bytes
                                )

                                output_buffer_size = (
                                    result["output_util"] * output_factor * batch_size_postprocessing
                                )
                                weight_buffer_size = (
                                    result["weight_util"] * weight_buf_factor
                                )
                                # It requires both input and output buffer size as the mapping is flexible
                                buffer_size = (
                                    weight_buffer_size
                                    + input_tensor_size
                                    + output_buffer_size
                                )
                                fused_results["interlayer_weight_util"] = [
                                    weight_buffer_size
                                ]
                                fused_results["max_buf_size"] = buffer_size
                                fused_results["M1"] = ""
                                results.append(fused_results)
                            if (
                                prob_idx != len(pair) - 1
                            ):  # last layer cannot skip output accesses
                                # print('skip outputs')
                                fused_results = {
                                    "orig_accesses": 0,
                                    "fused_accesses": 0,
                                    "max_buf_size": None,
                                    "mapping": result["mapping"] + "|",
                                }
                                fused_results["fused_accesses"] = (
                                    result["weight_accesses"] * weight_access_factor
                                    + result["input_accesses"]
                                    * input_factor
                                    * batch_size_postprocessing
                                )
                                fused_results["orig_accesses"] = fused_results[
                                    "fused_accesses"
                                ]
                                assert (
                                    fused_results["orig_accesses"]
                                    >= (weight_tensor_size + input_tensor_size ) / data_bytes
                                )

                                input_buffer_size = (
                                    result["input_util"] * input_factor * batch_size_postprocessing
                                )
                                weight_buffer_size = (
                                    result["weight_util"] * weight_buf_factor
                                )
                                # It requires both input and output buffer size as the mapping is flexible
                                buffer_size = (
                                    weight_buffer_size
                                    + input_buffer_size
                                    + output_tensor_size
                                )
                                fused_results["interlayer_weight_util"] = [
                                    weight_buffer_size
                                ]
                                fused_results["max_buf_size"] = buffer_size
                                fused_results["M1"] = ""
                                results.append(fused_results)

                    final_fused_results = {
                        "orig_accesses": [],
                        "fused_accesses": [],
                        "max_buf_size": [],
                        "interlayer_weight_util": [],
                        "M1": [],
                        "mapping": [],
                    }
                    for result in results:
                        for kk in final_fused_results.keys():
                            final_fused_results[kk].append(result[kk])

                    df = pd.DataFrame.from_dict(final_fused_results)

                    # logger.info(f'Optimal Accesses: {optimal_accesses_sub_chain}, Optimal Accesses Fused: {optimal_accesses_sub_chain_fused}')
                    output_csv = (
                        output_subdir
                        / f"{input_format}_{chain_idx}_subchain_{pair_idx}.csv"
                    )

                    df.to_csv(output_csv, index=False)
                    df = df[
                        [
                            "max_buf_size",
                            "orig_accesses",
                            "fused_accesses",
                            "interlayer_weight_util",
                            "M1",
                            "mapping",
                        ]
                    ]
                    df = df.sort_values(by=["max_buf_size"])

                    min_accesses_val = None
                    min_indices = []
                    for i, row in df.iterrows():
                        cur_val = row["orig_accesses"]
                        if min_accesses_val is None:
                            min_accesses_val = cur_val
                            min_indices.append(i)
                        else:
                            if cur_val < min_accesses_val:
                                min_accesses_val = cur_val
                                min_indices.append(i)
                    if keep_one_best_entry_across_buf:
                        df = df.loc[min_indices]

                    logger.info(f"Chain{chain_idx} Subchain{pair_idx} DF: \n{df}")
                    logger.info(
                        f"Chain{chain_idx} Subchain{pair_idx} DF total: \n{df_total_fused}"
                    )

                    # Fuse among subchains that has reduction that breaks it
                    if df_total_fused is not None:
                        df_total_fused = utils.merge_df(df_total_fused, df, m_value="")
                    else:
                        # if m_order = 0
                        df = df.sort_values(by=["max_buf_size"])
                        df_total_fused = df

            if df_total_fused is not None:
                # df_total_fused['orig_accesses'] = df_total_fused['orig_accesses'] # / 2 # added twice
                df_total_fused["reduction_rate"] = (
                    df_total_fused["orig_accesses"] - df_total_fused["fused_accesses"]
                ) / df_total_fused["orig_accesses"]
                df_total_fused["orig_accesses_cummin"] = df_total_fused[
                    "orig_accesses"
                ].cummin()
                df_total_fused["fused_accesses_cummin"] = df_total_fused[
                    "fused_accesses"
                ].cummin()
                df_total_fused["reduction_rate_cummin"] = (
                    df_total_fused["orig_accesses_cummin"]
                    - df_total_fused["fused_accesses_cummin"]
                ) / df_total_fused["orig_accesses_cummin"]
                # df_total_fused['reduction_rate_cummax'] = df['reduction_rate'].cummax()
                # df_util = df_total_fused[['interlayer_weight_util']]
                # print(f'df_util: {df_util}')

                logger.info(f"POST Chain{chain_idx} \n{df_total_fused}")

                df_total_fused = df_total_fused.sort_values(by=["max_buf_size"])
                df_total_fused = df_total_fused.reset_index(drop=True)

                if keep_one_best_entry_across_buf:
                    min_dram_accesses = None
                    max_index = []
                    for i, row in df_total_fused.iterrows():
                        cur_val = row[1]
                        if min_dram_accesses is None:
                            min_dram_accesses = cur_val
                            max_index.append(i)
                        else:
                            if cur_val < min_dram_accesses:
                                min_dram_accesses = cur_val
                                max_index.append(i)
                    df_total_fused = df_total_fused.loc[max_index]

                post_output_csv = (
                    f"{input_format}{fusion_str}_{chain_idx}_scheme{scheme}.csv"
                )
                abs_output_csv = cur_dir / post_output_csv
                logger.info(f"FINAL Chain{chain_idx}\n {df_total_fused}")
                logger.info(f"Output File: {abs_output_csv}")

                df_total_fused.to_csv(abs_output_csv, index=False)


if __name__ == "__main__":

    def wrapper(*args, **kwargs):
        logger.info(f"Arguments were: args={args}, kwargs={kwargs}")
        return process_untiled_fusion(*args, **kwargs)

    fire.Fire(process_untiled_fusion)
