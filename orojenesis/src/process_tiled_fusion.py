#!/usr/bin/env python3
import pathlib
import math
import fire
import numpy as np
import pandas as pd
import networkx as nx
from collections import OrderedDict
from . import utils

logger = utils.logger


def process_tiled_fusion(
    workload_dir,
    output_dir,
    model_name,
    input_format="chain",
    num_heads=1,
    batch=1,
    constraint_config="_relax_io_kn",
    scheme=2,
    matmul_only=True,
    ar="",
    nored="nored",
    spatial_factor=None,
    eval_slices=False,
    full_mapspace=False,
    arch_prefix="",
    data_bytes=2,
):
    """Processes tiled fusion for multi-einsum workloads with various mapping templates.

    This function processes the tiled fusion of deep learning model layers, optimizing them based on specified
    constraints, configurations, and evaluation modes. It's designed to optimize model inference in tiled architectures,
    supporting various input formats, constraint configurations, and enabling detailed architectural and operational analysis.

    Args:
        workload_dir (str): Directory containing the workloads to be processed.
        output_dir (str): Directory where the output will be stored.
        model_name (str, optional): Name of the model. Defaults to 'gpt3-6.7b'.
        input_format (str, optional): Format of the input workload. Options are 'chain' and 'opt_schedules_mm',
            with 'chain' keeping all fusible layers, and 'opt_schedules_mm' keeping a schedule with unfusible layers.
            Defaults to 'opt_schedules_mm'.
        constraint_config (str, optional): Configuration for the constraints. '_relax_io' allows partial k for the
            1st layer and partial n for the last layer, '_relax_io_kn' allows partial k and n for the 1st layer,
            and partial k (same as previous partial n) for the 2nd layer and partial n for the last layer. Otherwise,
            no partial k and n. Defaults to '_relax_io_kn'.
        scheme (int, optional): Scheme number for processing. Defaults to 2.
        num_heads (int, optional): Number of heads for post processing. Defaults to 1.
        batch (int, optional): Batch size for post processing. Defaults to 1.
        matmul_only (bool, optional): If True, only considers matrix multiplication operations. Defaults to True.
        ar (str, optional): Additional argument string for further customization. Defaults to ''.
        nored (str, optional): Specifies reduction behavior. Defaults to 'nored'.
        spatial_factor (int, optional): Factor for spatial partitioning. Defaults to None.
        eval_slices (bool, optional): If True, enables segmentation analysis. Defaults to False.
        full_mapspace (bool, optional): If True, evaluates the full map space. Defaults to False.
        arch_prefix (str, optional): Prefix for the architecture. Defaults to ''.
        data_bytes (int, optional): Number of bytes for each data element. Defaults to 2.

    Returns:
        None. The function processes the workload and saves the results in the specified output directory.

    Note:
        The `input_format` and `constraint_config` parameters provide detailed control over how model layers are fused
        and optimized, which can significantly affect the efficiency of the generated schedule and overall computational performance.
    """

    if arch_prefix == "simba_":
        total_levels = 6
        weight_level = 2
        input_level = 4
        output_level = 4
    else:
        total_levels = 3
        weight_level = 0
        input_level = 1
        output_level = 1
    if full_mapspace:
        oaves_file = "timeloop-mapper.oaves.csv"
    else:
        oaves_file = "oaves.csv"

    ar = "autoregressive_" if ar else ""
    nored = f"_{nored}" if nored != "" else ""
    batch = f"_b{batch}" if batch != "" else ""
    matmul = "_matmul" if matmul_only else ""
    heads = f"_h{num_heads}"
    spatial = f"_s{spatial_factor}" if spatial_factor is not None else ""
    fusion_str = "_fused"

    if batch != "":
        bmm_weight_loads = int(batch.split("b")[-1])
    else:
        raise ValueError("Invalid batch value!")

    for relax_io in [constraint_config]:

        graph_dir = pathlib.Path(
            f"{workload_dir}/{model_name}_graph{batch}{ar}{matmul}{nored}"
        )
        graph_dir_nc = pathlib.Path(
            f"{workload_dir}/{model_name}_graph{batch}{ar}{nored}"
        )

        layers_dict = utils.parse_yaml(graph_dir / "layers_dict.yaml")
        layers_dict_nc = utils.parse_yaml(graph_dir_nc / "layers_dict.yaml")
        if input_format == "chain":
            chains = utils.parse_yaml(graph_dir / "chains.yaml")
        else:
            chains = utils.parse_yaml(graph_dir / f"{input_format}.yaml")

        chain_tuples = utils.get_valid_chains(chains, layers_dict, graph_dir)

        chains = [chain_tuple[1] for chain_tuple in chain_tuples]

        P_prob = utils.get_prob_M_dim(chains, graph_dir, layers_dict)
        P_prob_log = int(np.log2(P_prob))

        if eval_slices:
            chain_slices = []
            for chain_idx, chain in chain_tuples:
                chain_slice = utils.parse_yaml(
                    graph_dir / f"{input_format}{chain_idx}_slices.yaml"
                )
                chain_slices.append(chain_slice)

        cur_dir = pathlib.Path(
            f"{output_dir}/{arch_prefix}{model_name}/scheme{scheme}{heads}{batch}{spatial}{ar}{matmul}{nored}{relax_io}"
        )
        cur_dir.mkdir(exist_ok=True, parents=True)

        def process_chain(output_dir, chains, batch, scheme, cur_dir, output_prefix):

            all_dfs = []
            for chain_idx, subchain in enumerate(chains):
                post_output_csv = f"{output_prefix}{chain_idx}_scheme{scheme}.csv"
                abs_output_csv = cur_dir / post_output_csv
                if abs_output_csv.exists():
                    df_total_fused = pd.read_csv(abs_output_csv, index_col=None)
                    if "slice_idx" not in df_total_fused.columns:
                        df_total_fused["slice_idx"] = f"{chain_idx}"
                        df_total_fused["slice"] = f"{subchain}"
                        df_total_fused.to_csv(abs_output_csv, index=False)
                    all_dfs.append(df_total_fused)

            df_total_fused_arr = [None] * len(chains)
            df_diff_m = []

            output_dir_opt = pathlib.Path(
                f"{output_dir}/{arch_prefix}{model_name}/oaves_chain_2levels_outputs{batch}{spatial}_opt"
            )
            for m_order in range(P_prob_log + 1):
                m_value = 2**m_order

                output_dir_ffmt = pathlib.Path(
                    f"{output_dir}/{arch_prefix}{model_name}/oaves_chain_2levels_outputs{batch}{spatial}_p{m_value}"
                )
                output_dir_allowk = pathlib.Path(
                    f"{output_dir}/{arch_prefix}{model_name}/oaves_chain_2levels_outputs{batch}{spatial}_allowk_p{m_value}"
                )
                output_dir_allown = pathlib.Path(
                    f"{output_dir}/{arch_prefix}{model_name}/oaves_chain_2levels_outputs{batch}{spatial}_allown_p{m_value}"
                )
                output_dir_allowkn = pathlib.Path(
                    f"{output_dir}/{arch_prefix}{model_name}/oaves_chain_2levels_outputs{batch}{spatial}_allowkn_p{m_value}"
                )
                output_dir_allownk = pathlib.Path(
                    f"{output_dir}/{arch_prefix}{model_name}/oaves_chain_2levels_outputs{batch}{spatial}_allownk_p{m_value}"
                )

                output_dir_allowk_flash = pathlib.Path(
                    f"{output_dir}/{arch_prefix}{model_name}/oaves_chain_2levels_outputs{batch}{spatial}_allowk_flash_p{m_value}"
                )
                output_dir_allown_flash = pathlib.Path(
                    f"{output_dir}/{arch_prefix}{model_name}/oaves_chain_2levels_outputs{batch}{spatial}_allown_flash_p{m_value}"
                )

                df_diff_chains = []

                # continue
                for chain_idx, subchain in enumerate(chains):
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
                            if "prob_" in layers_dict[prob]:
                                dims = layers_dict[prob].replace("prob_", "").split("_")
                            elif "red_" in layers_dict[prob]:
                                dims = layers_dict[prob].replace("red_", "").split("_")
                            elif "ew_" in layers_dict[prob]:
                                dims = layers_dict[prob].replace("ew_", "").split("_")
                            else:
                                raise (f"Unknown op {layers_dict[prob]}!")

                            dims = [int(dim) for dim in dims]
                            subchain_dims.append(dims)

                        for prob_idx, prob in enumerate(pair):
                            dims = subchain_dims[prob_idx]
                            logger.info(
                                f"Layer{prob_idx}: {prob} {layers_dict[prob]} {dims}"
                            )

                            if "prob_" in layers_dict[prob]:
                                min_weight_accesses = dims[1] * dims[2]
                                min_input_accesses = dims[0] * dims[1]
                                min_output_accesses = dims[0] * dims[2]
                            elif "red_" in layers_dict[prob]:
                                min_weight_accesses = 0  # dims[1]*dims[2]
                                min_input_accesses = dims[0] * dims[1]
                                min_output_accesses = dims[0]
                            elif "ew_" in layers_dict[prob]:
                                min_weight_accesses = 0  # dims[1]*dims[2]
                                min_input_accesses = dims[0] * dims[1]
                                min_output_accesses = dims[0]

                            buffer_all = True
                            (
                                input_factor,
                                output_factor,
                                weight_access_factor,
                                weight_buf_factor,
                            ) = utils.compute_num_head_factors(
                                prob, num_heads, buffer_all, bmm_weight_loads
                            )
                            optimal_accesses_sub_chain_fused += (
                                min_weight_accesses * weight_access_factor
                            )
                            if prob_idx == 0:
                                optimal_accesses_sub_chain_fused += (
                                    min_input_accesses * input_factor
                                )
                            if prob_idx == len(pair) - 1:
                                optimal_accesses_sub_chain_fused += (
                                    min_output_accesses * output_factor
                                )
                            optimal_accesses_sub_chain += (
                                min_weight_accesses * weight_access_factor
                                + min_input_accesses * input_factor
                                + min_output_accesses * output_factor
                            )
                            logger.info(
                                f"Optimal Layer Accesses: {optimal_accesses_sub_chain}"
                            )
                            logger.info(
                                f"Optimal Layer Accesses Fused: {optimal_accesses_sub_chain_fused}"
                            )
                        chain_dims.append(subchain_dims)

                    # Process each subchain
                    op_no_partialn = ["mm_qk", "mm_proj_final", "mm_fc2", "bmm_qk"]

                    for pair_idx, pair in enumerate(subchain):
                        logger.info(f"Chain{chain_idx} Subchain{pair_idx}: {pair}")
                        subchain_dfs = []
                        for prob_idx, prob in enumerate(pair):
                            if relax_io == "_relax_io":  # flat
                                if prob_idx == 0 and prob_idx == len(pair) - 1:
                                    if prob in ['mm_qk', 'mm_qkv']:
                                        bmm_prob = 'b' + prob
                                    else:
                                        bmm_prob = prob
                                    oaves_csv = (
                                        output_dir_opt
                                        / layers_dict_nc[bmm_prob]
                                        / oaves_file
                                    )
                                elif prob_idx == 0:
                                    oaves_csv = (
                                        output_dir_allowk
                                        / layers_dict[prob]
                                        / oaves_file
                                    )
                                elif prob_idx == len(pair) - 1:
                                    oaves_csv = (
                                        output_dir_allown
                                        / layers_dict[prob]
                                        / oaves_file
                                    )
                                else:
                                    oaves_csv = (
                                        output_dir_ffmt / layers_dict[prob] / oaves_file
                                    )

                            elif relax_io == "_flash":
                                if prob_idx == 0 and prob_idx == len(pair) - 1:
                                    if prob in ['mm_qk', 'mm_qkv']:
                                        bmm_prob = 'b' + prob
                                    else:
                                        bmm_prob = prob
                                    oaves_csv = (
                                        output_dir_opt
                                        / layers_dict_nc[bmm_prob]
                                        / oaves_file
                                    )
                                elif prob_idx == 0:
                                    oaves_csv = (
                                        output_dir_allown_flash
                                        / layers_dict[prob]
                                        / oaves_file
                                    )
                                elif prob_idx == len(pair) - 1:
                                    oaves_csv = (
                                        output_dir_allowk_flash
                                        / layers_dict[prob]
                                        / oaves_file
                                    )
                                else:
                                    oaves_csv = (
                                        output_dir_ffmt / layers_dict[prob] / oaves_file
                                    )

                            elif relax_io == "_relax_io_kn" or relax_io == "_relax_io_kn_flash":
                                if prob_idx == 1 and prob_idx == len(pair) - 1:
                                    oaves_csv = (
                                        output_dir_allownk
                                        / layers_dict[prob]
                                        / oaves_file
                                    )
                                elif (
                                    prob_idx == 0
                                ):  # including and prob_idx == len(pair) - 1

                                    # Default constraint in FFMT for idx=0
                                    oaves_csv = (
                                        output_dir_allowkn
                                        / layers_dict[prob]
                                        / oaves_file
                                    )

                                    # If a reduction is present in the following op
                                    # partial n shouldn't be allowed
                                    if (
                                        prob in op_no_partialn
                                        and prob_idx != len(pair) - 1
                                        and "attn-block" not in model_name
                                    ):
                                        oaves_csv = (
                                            output_dir_allowk
                                            / layers_dict[prob]
                                            / oaves_file
                                        )

                                elif prob_idx == 1:
                                    oaves_csv = (
                                        output_dir_allowk
                                        / layers_dict[prob]
                                        / oaves_file
                                    )
                                elif prob_idx == len(pair) - 1:
                                    oaves_csv = (
                                        output_dir_allown
                                        / layers_dict[prob]
                                        / oaves_file
                                    )
                                else:
                                    oaves_csv = (
                                        output_dir_ffmt / layers_dict[prob] / oaves_file
                                    )
                                if relax_io == "_relax_io_kn_flash":
                                    if len(pair) == 2:
                                        if prob_idx == 0:
                                            oaves_csv = (
                                                output_dir_allown_flash
                                                / layers_dict[prob]
                                                / oaves_file
                                            )
                                        elif prob_idx == 1:
                                            oaves_csv = (
                                                output_dir_allowk_flash
                                                / layers_dict[prob]
                                                / oaves_file
                                            )
                                        else:
                                            raise ValueError("Invalide Einsum position!")

                            elif relax_io == "":
                                pass
                            else:
                                raise (
                                    f'Invalid relax_io option {relax_io}! Only "" "_relax_io" "_relax_io_kn" is allowed.'
                                )

                            if len(pair) == 1:
                                if prob in ['mm_qk', 'mm_qkv']:
                                    bmm_prob = 'b' + prob
                                else:
                                    bmm_prob = prob

                                oaves_csv = (
                                    output_dir_opt / layers_dict_nc[bmm_prob] / oaves_file
                                )

                            logger.info(f"Parse CSV: {oaves_csv}")
                            try:
                                df = pd.read_csv(oaves_csv, header=None)
                                df = df.reset_index(drop=True)


                                if len(pair) != 1:
                                    (
                                        input_factor,
                                        output_factor,
                                        weight_access_factor,
                                        weight_buf_factor,
                                    ) = utils.compute_num_head_factors(
                                        prob, num_heads, buffer_all=False, batch=bmm_weight_loads
                                    )

                                    _, tensor_names_to_index = utils.map_index_to_tensor_names(total_levels, weight_level, input_level, output_level)


                                    utils.update_df_metric(df, tensor_names_to_index, 'weight_util', weight_buf_factor)
                                    utils.update_df_metric(df, tensor_names_to_index, 'weight_accesses', weight_access_factor)
                                    utils.update_df_metric(df, tensor_names_to_index, 'input_util', input_factor)
                                    utils.update_df_metric(df, tensor_names_to_index, 'input_accesses', input_factor)
                                    utils.update_df_metric(df, tensor_names_to_index, 'output_util', output_factor)
                                    utils.update_df_metric(df, tensor_names_to_index, 'output_accesses', output_factor)

                                    utils.update_df_metric_sum(df, tensor_names_to_index, 'buf_size', ['weight_util', 'input_util', 'output_util'])
                                    utils.update_df_metric_sum(df, tensor_names_to_index, 'total_accesses', ['weight_accesses', 'input_accesses', 'output_accesses'])

                            except:
                                raise
                            subchain_dfs.append(df)

                        total_accesses = None

                        # For each output buf size, start processing multiple chain
                        results = []

                        G = nx.DiGraph()
                        nodes = OrderedDict()
                        # Adjecency list to represent the graph
                        adj = {}
                        source_nodes = []
                        end_nodes = []
                        logger.info(
                            f"Number of Layers in the Subchain: {len(subchain_dfs)}"
                        )

                        output_utils = []
                        # Create a graph for each init target buf size
                        for subseq_df_idx, subseq_df in enumerate(subchain_dfs):
                            logger.debug(f"Subchain layer idx: {subseq_df_idx}")

                            # verification
                            if subseq_df_idx == 0:
                                # iterate through all rows

                                logger.debug(f"BEFORE pruning {len(subseq_df)}")
                                subseq_df = utils.df_one_best_entry_across_buf(
                                    subseq_df, 0, 2
                                )
                                logger.debug(f"AFTER pruning {len(subseq_df)}")

                                for row_idx, (
                                    matched_row_idx,
                                    matched_row,
                                ) in enumerate(subseq_df.iterrows()):
                                    result = utils.row_to_result_dict(
                                        matched_row,
                                        total_levels=total_levels,
                                        weight_level=weight_level,
                                        input_level=input_level,
                                        output_level=input_level,
                                    )
                                    output_utils.append(result["output_util"])
                                    key = (subseq_df_idx, row_idx)
                                    nodes[key] = result
                                    adj[key] = []
                                    source_nodes.append(key)
                                    if subseq_df_idx == len(subchain_dfs) - 1:
                                        end_nodes.append(key)
                                logger.debug(f"adj 0 {adj.keys()}")
                                logger.debug(
                                    f"Subchain layer {subseq_df_idx} Output Utils: {output_utils}"
                                )
                            else:
                                prev_output_utils = output_utils[:]
                                output_utils = []
                                logger.debug(
                                    f"Subchain layer {subseq_df_idx} Prev Output Utils: {prev_output_utils}"
                                )

                                for prev_output_util_idx, prev_output_util in enumerate(
                                    prev_output_utils
                                ):
                                    input_util_idx = 6 + input_level * 3 + 1
                                    matched_rows = subseq_df.loc[
                                        subseq_df[input_util_idx] == prev_output_util
                                    ]

                                    logger.debug(f"BEFORE pruning {len(matched_rows)}")
                                    # Need to prune the matched rows
                                    matched_rows = utils.df_one_best_entry_across_buf(
                                        matched_rows, 0, 2
                                    )
                                    logger.debug(f"AFTER pruning {len(matched_rows)}")

                                    logger.debug(
                                        f"Prev Output Utils {prev_output_util_idx} Matched Input Rows {len(matched_rows)}"
                                    )
                                    for _, matched_row in matched_rows.iterrows():
                                        result = utils.row_to_result_dict(
                                            matched_row,
                                            total_levels=total_levels,
                                            weight_level=weight_level,
                                            input_level=input_level,
                                            output_level=input_level,
                                        )

                                        matched_row_idx = len(output_utils)
                                        output_utils.append(result["output_util"])

                                        key = (subseq_df_idx, matched_row_idx)
                                        nodes[key] = result
                                        adj[key] = []
                                        prev_key = (
                                            subseq_df_idx - 1,
                                            prev_output_util_idx,
                                        )
                                        adj[prev_key].append(key)
                                        if subseq_df_idx == len(subchain_dfs) - 1:
                                            end_nodes.append(key)

                        logger.info(
                            f"Chain{chain_idx} Subchain{pair_idx} Start Processing Graphs!"
                        )
                        logger.info(
                            f"Adj list: {adj}, Source nodes: {source_nodes}, End nodes: {end_nodes}"
                        )
                        G = nx.DiGraph()
                        for k, v in adj.items():
                            G.add_node(k)
                            for node in v:
                                G.add_edge(k, node)

                        paths = []
                        if len(pair) == 1:
                            for source_node in source_nodes:
                                assert source_node in end_nodes
                                paths.append([source_node])

                        for source_node in source_nodes:
                            subpaths = list(
                                nx.all_simple_paths(
                                    G, source=source_node, target=end_nodes
                                )
                            )
                            paths.extend(subpaths)

                        logger.info(f"\tNum Source Nodes: {len(source_nodes)}")
                        logger.info(f"\tNum Paths: {len(paths)}")
                        interlayer_weight_buf = 0

                        # Process the path for one single target buffer utilization
                        for path_idx, path in enumerate(paths):
                            path = list(path)
                            optimal_accesses_sub_chain = 0
                            optimal_accesses_sub_chain_fused = 0

                            logger.info(f"Path{path_idx}")

                            fused_results = {
                                "orig_accesses": 0,
                                "fused_accesses": 0,
                                "max_buf_size": None,
                                "mapping": "",
                            }

                            path_len = sum(1 for _ in path)
                            interlayer_weight_util = []

                            scheme2_subpath_info = []
                            scheme2_buffer_sizes = []
                            sub_buffer_size = 0

                            prev_buffer_all = False
                            for node_idx, node_key in enumerate(path):
                                prob = pair[node_idx]
                                dims = chain_dims[pair_idx][node_idx]
                                logger.debug(
                                    f"Path{path_idx} Layer{node_idx}: {node_key} {prob} {dims}"
                                )
                                min_weight_accesses = dims[1] * dims[2]
                                min_input_accesses = dims[0] * dims[1]
                                min_output_accesses = dims[0] * dims[2]

                                fused_results["mapping"] += (
                                    nodes[node_key]["mapping"] + "|"
                                )
                                # if we buffer all the weights, we should multiply the weight utilization by h, but not multiply the dram access. if we don't buffer all weights, we should multiply the dram accesses by h, and also the buffer by h
                                weight_util = nodes[node_key]["weight_util"]

                                if (
                                    weight_util != min_weight_accesses * data_bytes
                                    and weight_util != data_bytes
                                ):
                                    oaves_csv = (
                                        output_dir_ffmt / layers_dict[prob] / oaves_file
                                    )
                                    logger.error(dims)
                                    logger.error(f"weight_util: {weight_util}")
                                    logger.error(
                                        f"min_weight_accesses: {min_weight_accesses * data_bytes}"
                                    )
                                    # raise("This post processing script only works for buffer all weights or no weights!")

                                buffer_all = (
                                    weight_util == min_weight_accesses * data_bytes
                                )

                                (
                                    input_factor,
                                    output_factor,
                                    weight_access_factor,
                                    weight_buf_factor,
                                ) = utils.compute_num_head_factors(
                                    prob, num_heads,
                                    (buffer_all or prob in ["mm_qk", "mm_qkv"]) and len(pair) != 1,
                                    bmm_weight_loads
                                )

                                min_total_accesses = (
                                    min_weight_accesses * weight_access_factor
                                    + min_input_accesses * input_factor
                                    + min_output_accesses * output_factor
                                )

                                if node_idx == 0:
                                    optimal_accesses_sub_chain_fused += (
                                        min_input_accesses * input_factor
                                    )
                                if node_idx == path_len - 1:
                                    optimal_accesses_sub_chain_fused += (
                                        min_output_accesses * output_factor
                                    )
                                optimal_accesses_sub_chain_fused += (
                                    min_weight_accesses * weight_access_factor
                                )
                                optimal_accesses_sub_chain += (
                                    min_weight_accesses * weight_access_factor
                                    + min_input_accesses * input_factor
                                    + min_output_accesses * output_factor
                                )

                                logger.debug(
                                    f"\tMin Total Accesses: {min_total_accesses}"
                                )

                                # fused_results['orig_accesses'] += nodes[node_key]['total_accesses']
                                fused_results["orig_accesses"] += (
                                    nodes[node_key]["total_accesses"]
                                )

                                # special handling for 2nd input to the bmm, which is the weight in each timeloop op

                                bmm_weight_access_factor = 1
                                if len(pair) > 1:
                                    if prob in ["mm_qk", "mm_qkv"]:
                                        if buffer_all:
                                            bmm_weight_access_factor = bmm_weight_loads
                                        elif m_value < bmm_weight_loads:
                                            bmm_weight_access_factor = bmm_weight_loads // m_value

                                logger.info(f'RESULT buffer_all: {buffer_all} m_value: {m_value} bmm_weight_loads: {bmm_weight_loads}  bmm_weight_access_factor: {bmm_weight_access_factor}')
                                bmm_weight_access = nodes[node_key]["weight_accesses"] * bmm_weight_access_factor

                                fused_results["fused_accesses"] += bmm_weight_access

                                if node_idx == 0:
                                    fused_results["fused_accesses"] += (
                                        nodes[node_key]["input_accesses"]
                                    )
                                if node_idx == path_len - 1:
                                    fused_results["fused_accesses"] += (
                                        nodes[node_key]["output_accesses"]
                                    )
                                    logger.debug(
                                        f"Path{path_idx} Fused Accesses: {fused_results['fused_accesses']}"
                                    )
                                # print(nodes[node_key])
                                weight_buffer_size = nodes[node_key]["weight_util"]
                                input_buffer_size = (
                                    nodes[node_key]["input_util"]
                                )
                                output_buffer_size = (
                                    nodes[node_key]["output_util"]
                                )

                                if scheme in [1, 2]:
                                    weight_buffer_size = (
                                        nodes[node_key]["weight_util"]
                                        * weight_buf_factor
                                    )
                                    input_buffer_size = (
                                        nodes[node_key]["input_util"]
                                    )
                                    output_buffer_size = (
                                        nodes[node_key]["output_util"]
                                    )

                                logger.debug(
                                    f"Path{path_idx} Layer{node_idx}: buffer_all={buffer_all} prev_buffer_all={prev_buffer_all}"
                                )

                                if scheme == 1 or scheme == 3:
                                    if buffer_all:
                                        interlayer_weight_util.append(
                                            weight_buffer_size
                                        )
                                        buffer_size = (
                                            input_buffer_size + output_buffer_size
                                        )
                                    else:
                                        interlayer_weight_util.append(0)
                                        buffer_size = (
                                            weight_buffer_size
                                            + input_buffer_size
                                            + output_buffer_size
                                        )
                                    if fused_results["max_buf_size"] is not None:
                                        if fused_results["max_buf_size"] < buffer_size:
                                            fused_results["max_buf_size"] = buffer_size
                                    else:
                                        fused_results["max_buf_size"] = buffer_size
                                elif scheme == 2 or scheme == 4:
                                    if buffer_all:
                                        interlayer_weight_util.append(
                                            weight_buffer_size
                                        )

                                    next_buffer_all = False  # if node_idx == path_len -1, we are not buffer all weight and should compare current IO
                                    if node_idx + 1 < path_len:
                                        next_node_key = path[node_idx + 1]
                                        next_dims = chain_dims[pair_idx][node_idx + 1]
                                        next_weight_util = nodes[next_node_key][
                                            "weight_util"
                                        ]
                                        next_min_weight_accesses = (
                                            next_dims[1] * next_dims[2]
                                        )
                                        next_buffer_all = (
                                            next_weight_util
                                            == next_min_weight_accesses * data_bytes
                                        )
                                    if buffer_all:
                                        # first buffer all
                                        if not prev_buffer_all:
                                            # initialize scheme2 first input buffer and weight buffer requirements
                                            first_input_buffer_size = input_buffer_size
                                            sub_weight_buffer_size = 0

                                            # initialize scheme2 buffer requirements
                                            scheme2_buffer_sizes.append(0)

                                        # calculate the max IO buffer size w/out weight, first input and last output buffer size
                                        buffer_size = 0
                                        if prev_buffer_all:
                                            buffer_size += (
                                                dims[1] * input_factor * data_bytes
                                            )  # K*1
                                        if next_buffer_all:
                                            buffer_size += (
                                                dims[2] * output_factor * data_bytes
                                            )  # N*1
                                        if buffer_size > scheme2_buffer_sizes[-1]:
                                            scheme2_buffer_sizes[-1] = buffer_size

                                        # always add to the weights if buffer all
                                        sub_weight_buffer_size += weight_buffer_size

                                        # last buffer all
                                        if not next_buffer_all:
                                            last_output_buffer_size = output_buffer_size
                                            scheme2_subpath_info.append(
                                                (
                                                    scheme2_buffer_sizes[-1],
                                                    sub_weight_buffer_size,
                                                    first_input_buffer_size,
                                                    last_output_buffer_size,
                                                )
                                            )
                                            # an optimization is to take the max of IO buffer, so output can override input buffer
                                            # scheme2_buffer_sizes[-1] +=  sub_weight_buffer_size + max(first_input_buffer_size, last_output_buffer_size)
                                            scheme2_buffer_sizes[
                                                -1
                                            ] += sub_weight_buffer_size + first_input_buffer_size + last_output_buffer_size
                                    else:
                                        buffer_size = (
                                            weight_buffer_size
                                            + input_buffer_size
                                            + output_buffer_size
                                        )
                                        scheme2_buffer_sizes.append(buffer_size)
                                        interlayer_weight_util.append(0)
                                else:
                                    raise ("Invalid scheme!")

                                # add up all interlayer weights
                                if node_idx == path_len - 1:
                                    fused_results["interlayer_weight_util"] = (
                                        interlayer_weight_util
                                    )
                                    fused_results["M1"] = m_value
                                    if scheme == 1 or scheme == 3:
                                        total_interlayer_weight_buf = sum(
                                            interlayer_weight_util
                                        )
                                        logger.info(
                                            f"Interlayer Weight Util: {interlayer_weight_util} -> {total_interlayer_weight_buf}"
                                        )
                                        fused_results[
                                            "max_buf_size"
                                        ] += total_interlayer_weight_buf
                                    elif scheme == 2 or scheme == 4:
                                        logger.info(
                                            f"Scheme2 buffer size -> {scheme2_buffer_sizes}"
                                        )
                                        fused_results["max_buf_size"] = max(
                                            scheme2_buffer_sizes
                                        )
                                    prev_buffer_all = False
                                prev_buffer_all = buffer_all

                            logger.info(f"Fused Results: {fused_results}")
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

                        output_csv = (
                            output_dir_ffmt
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
                        logger.info(f"Chain{chain_idx} Subchain{pair_idx} DF: \n{df}")
                        logger.info(
                            f"Chain{chain_idx} Subchain{pair_idx} DF total: \n{df_total_fused}"
                        )

                        # Fuse among subchains that has reduction that breaks it
                        if df_total_fused is not None:
                            df_total_fused = utils.merge_df(df_total_fused, df, m_value)
                        else:
                            # if m_order = 0
                            df = df.sort_values(by=["max_buf_size"])
                            df_total_fused = df

                    if df_total_fused is not None:
                        # df_total_fused['orig_accesses'] = df_total_fused['orig_accesses'] # / 2 # added twice
                        df_total_fused["reduction_rate"] = (
                            df_total_fused["orig_accesses"]
                            - df_total_fused["fused_accesses"]
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

                        logger.info(
                            f"POST Chain{chain_idx} M1={m_value} \n{df_total_fused}"
                        )

                        if m_order == 0:
                            df_total_fused_arr[chain_idx] = df_total_fused
                        else:
                            data = [df_total_fused_arr[chain_idx], df_total_fused]
                            df_total_fused_arr[chain_idx] = pd.concat(data)

                    df_diff_chains.append(df)
                    output_csv = f"{input_format}_M{m_value}_{output_prefix}{chain_idx}_scheme{scheme}.csv"
                    df_total_fused.to_csv(cur_dir / output_csv, index=False)
                df_diff_m.append(df_diff_chains)

            keep_one_best_entry_across_buf = True
            all_dfs = []
            for chain_idx, subchain in enumerate(chains):
                df_total_fused = df_total_fused_arr[chain_idx].sort_values(
                    by=["max_buf_size"]
                )
                df_total_fused = df_total_fused.reset_index(drop=True)

                if keep_one_best_entry_across_buf:
                    min_dram_accesses = None
                    max_index = []
                    for i, row in df_total_fused.iterrows():
                        cur_val = row["fused_accesses"]
                        if min_dram_accesses is None:
                            min_dram_accesses = cur_val
                            max_index.append(i)
                        else:
                            if cur_val < min_dram_accesses:
                                min_dram_accesses = cur_val
                                max_index.append(i)
                    df_total_fused = df_total_fused.loc[max_index]

                post_output_csv = f"{output_prefix}{chain_idx}_scheme{scheme}.csv"
                logger.info(f"FINAL Chain {chain_idx}\n {df_total_fused}")

                abs_output_csv = cur_dir / post_output_csv
                logger.info(f"Output File: {abs_output_csv}")

                df_total_fused["slice_idx"] = f"{chain_idx}"
                df_total_fused["slice"] = f"{subchain}"
                df_total_fused.to_csv(abs_output_csv, index=False)
                all_dfs.append(df_total_fused)
            return all_dfs

        if eval_slices:
            for chain_idx, chain_slice in enumerate(chain_slices):
                all_dfs = process_chain(
                    output_dir,
                    chain_slice,
                    batch,
                    scheme,
                    cur_dir,
                    f"{input_format}{fusion_str}_{chain_idx}_slice",
                )
                logger.info(f"Processed {len(all_dfs)} slices. Start Merging")
                print(len(all_dfs))
                all_slice_df = utils.merge_df_slices(all_dfs)
                post_output_csv = (
                    f"{input_format}{fusion_str}_{chain_idx}_slice_scheme{scheme}.csv"
                )
                abs_output_csv = cur_dir / post_output_csv
                all_slice_df.to_csv(abs_output_csv, index=False)
                logger.info(f"FINAL SLICE Chain {chain_idx}")
                logger.info(f"Output File: {abs_output_csv}")
        else:
            all_dfs = process_chain(
                output_dir,
                chains,
                batch,
                scheme,
                cur_dir,
                f"{input_format}{fusion_str}_",
            )
            logger.info(f"Processed {len(all_dfs)} Chains.")


if __name__ == "__main__":

    def wrapper(*args, **kwargs):
        print(f"Arguments were: args={args}, kwargs={kwargs}")
        return process_tiled_fusion(*args, **kwargs)

    fire.Fire(process_tiled_fusion)
