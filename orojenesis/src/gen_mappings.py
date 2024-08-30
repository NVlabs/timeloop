#!/usr/bin/env python3
import pathlib
import math
import fire
import numpy as np
import shutil

from . import utils

logger = utils.logger


def run_einsums(chain, output_file, mapping_yaml, arch_yaml, force_rerun=False):

    # Step 0: Check the Matmul operation is valid
    for einsum in chain:
        if einsum["type"] == "Matmul":
            if (
                einsum["prob_dims"]["N"] != 1
                or einsum["prob_dims"]["R"] != 1
                or einsum["prob_dims"]["S"] != 1
                or einsum["prob_dims"]["Q"] != 1
            ):
                raise ("Invalid problem specficiation!")

    # Step 1: Run Weld.
    for einsum in chain:
        rundir = pathlib.Path(einsum["rundir"])
        rundir.mkdir(parents=True, exist_ok=True)

        workload_yaml = rundir / "problem.yaml"

        #   - Generate the user defined problem dims
        if einsum["type"] == "Elementwise_2op" or einsum["type"] == "BMM":
            shutil.copy(einsum["prob_yaml"], workload_yaml)
        else:
            utils.GenProblemYAML(einsum["prob_dims"], workload_yaml)

        #   - Run the usual post-processing script
        preprocessed_csv = rundir / "orojenesis.csv"
        if force_rerun or not preprocessed_csv.exists():
            if einsum["type"] == "Elementwise_2op":
                utils.RunMapper(arch_yaml, workload_yaml, einsum["mapper"], rundir)
            elif einsum["type"] == "BMM":
                num_heads = einsum['num_heads']
                bmm_arch_yaml = rundir / "arch.yaml"
                arch = utils.parse_yaml(arch_yaml)
                arch['architecture']['subtree'][0]['subtree'][0]['name'] = f'PE [0..{num_heads-1}]'
                utils.store_yaml(bmm_arch_yaml, arch)

                bmm_mapper_yaml = rundir / "mapper.yaml"
                mapper_yaml = einsum["mapper"]
                mapper = utils.parse_yaml(mapper_yaml)
                for constraint in mapper["mapspace_constraints"]:
                    if constraint['type'] == 'spatial':
                        constraint['factors'] = f'H={num_heads}'

                utils.store_yaml(bmm_mapper_yaml, mapper)
                utils.RunMapper(bmm_arch_yaml, workload_yaml, bmm_mapper_yaml, rundir)
            else:
                utils.RunMapper(arch_yaml, workload_yaml, mapping_yaml, rundir)
            utils.RunOrojenesisPostProcessing(rundir)
        orojenesis_csv = rundir / "orojenesis.csv"
    logger.info(f"Finished generating for einsum: {einsum}")
    return


def gen_einsums(
    graph_dir,
    config_dir,
    output_subdir,
    mapping,
    mapping_ew,
    pair_chains,
    layers_dict,
    arch_yaml,
    matmul_only,
    num_heads=None,
    force_rerun=False,
    spatial_factor=None,
):
    output_subdir.mkdir(parents=True, exist_ok=True)
    mapper_file = output_subdir / "mapper.yaml"
    utils.store_yaml(mapper_file, mapping)
    mapper_ew_file = output_subdir / "mapper_ew.yaml"
    utils.store_yaml(mapper_ew_file, mapping_ew)

    orojenesis_input_chains = []
    for chain_idx, pair_chain in enumerate(pair_chains):
        for pair_idx, pair in enumerate(pair_chain):
            output_csv = output_subdir / f"chain_{chain_idx}_pair_{pair_idx}.csv"
            pair_chain = []
            for prob_idx, prob in enumerate(pair):
                prob_dict = {}
                prob_dict["rundir"] = str(output_subdir / layers_dict[prob])

                if "mm_" in prob:
                    if prob_idx == 1 and prob == "mm_qkv":
                        prob_dict["input"] = "Weights"
                    else:
                        prob_dict["input"] = "Inputs"
                    prob_dict["output"] = "Outputs"
                    prob_dict["type"] = "Matmul"
                    dims = layers_dict[prob].split("_")[1:]
                    dims = [int(dim) for dim in dims]
                    prob_dict["prob_dims"] = {
                        "R": 1,
                        "S": 1,
                        "P": dims[0],
                        "Q": 1,
                        "C": dims[1],
                        "K": dims[2],
                        "N": 1,
                    }
                elif "red_" in prob:
                    prob_dict["output"] = "Outputs"
                    prob_dict["type"] = "Matmul"
                    dims = layers_dict[prob].split("_")[1:]
                    dims = [int(dim) for dim in dims]
                    prob_dict["prob_dims"] = {
                        "R": 1,
                        "S": 1,
                        "P": dims[0],
                        "Q": 1,
                        "C": dims[1],
                        "K": 1,
                        "N": 1,
                    }
                elif "ew_" in prob:
                    prob_dict["output"] = "Outputs"
                    prob_dict["type"] = "Elementwise_2op"
                    dims = layers_dict[prob].split("_")[1:]
                    dims = [int(dim) for dim in dims]
                    prob_dict["prob_dims"] = {"P": dims[0], "C": dims[1]}
                    prob_dict["prob_yaml"] = graph_dir / f"{layers_dict[prob]}.yaml"
                    prob_dict["mapper"] = f"{config_dir}/mapper_ew.yaml"

                if not matmul_only:
                    if "bmm_" in prob:
                        if prob_idx == 1 and prob == "bmm_qkv":
                            prob_dict["input"] = "Weights"
                        else:
                            prob_dict["input"] = "Inputs"
                        prob_dict["output"] = "Outputs"
                        prob_dict["type"] = "BMM"
                        dims = layers_dict[prob].split("_")[1:]
                        dims = [int(dim) for dim in dims]
                        prob_dict["prob_dims"] = {
                            "M": dims[1],
                            "K": dims[2],
                            "N": dims[3],
                            "H": dims[0],
                        }
                        prob_dict["prob_yaml"] = graph_dir / f"{layers_dict[prob]}.yaml"
                        prob_dict["mapper"] = f"{config_dir}/mapper_bmm.yaml"
                        # We assume we have all heads  are executed in parallel
                        prob_dict["num_heads"] = num_heads

                pair_chain.append(prob_dict)

            run_einsums(
                pair_chain, output_csv, mapper_file, arch_yaml, force_rerun=force_rerun
            )
            # try:
            #     gen_chain(pair_chain, output_csv, mapper_file, arch_yaml)
            # except:
            #     raise Exception('Failed to generate mappings')


def gen_mappings(
    workload_dir,
    config_dir,
    output_dir,
    model_name="gpt3-6.7b",
    batch=16,
    num_heads=32,
    ffmt=True,
    ar="",
    nored="nored",
    spatial_factor=None,
    arch_prefix="",
    force_rerun=False,
):
    matmul_only = True
    matmul = "_matmul" if matmul_only else ""
    nored = f"_{nored}" if nored != "" else ""
    batch = f"_b{batch}" if batch != "" else ""
    spatial = f"_s{spatial_factor}" if spatial_factor is not None else ""

    mapping_postfice = ["", "_allown", "_allowk", "_allowkn", "_allownk", "_allowk_flash", "_allown_flash"]

    graph_dir = pathlib.Path(
        f"{workload_dir}/{model_name}_graph{batch}{ar}{matmul}{nored}"
    )

    graph_dir_nc = pathlib.Path(
        f"{workload_dir}/{model_name}_graph{batch}{ar}{nored}"
    )

    logger.info(f"Einsum chain definition is in: {graph_dir}")
    if arch_prefix != "":
        pair_chains = utils.parse_yaml(graph_dir / "pair_chains.yaml")
        pair_chains_nc = utils.parse_yaml(graph_dir_nc / "pair_chains.yaml")
    else:
        pair_chains = utils.parse_yaml(graph_dir / "opt_schedules.yaml")
        pair_chains_nc = utils.parse_yaml(graph_dir_nc / "opt_schedules.yaml")
    layers_dict = utils.parse_yaml(graph_dir / "layers_dict.yaml")
    layers_dict_nc = utils.parse_yaml(graph_dir_nc / "layers_dict.yaml")

    arch_yaml = f"{config_dir}/{arch_prefix}arch.yaml"

    # parse the first layer M/P dim
    P_prob = utils.get_prob_M_dim(pair_chains, graph_dir, layers_dict)
    P_prob_log = int(np.log2(P_prob))

    def update_mapping(
        mapping, memlevel, factor_str, permutation_str, mapping_type="temporal"
    ):
        entry_found = False
        for entry in mapping["mapspace_constraints"]:
            if entry["target"] == memlevel and entry["type"] == mapping_type:
                entry["factors"] = factor_str
                entry["permutation"] = permutation_str
                entry_found = True
        if not entry_found:
            entry = {
                "target": memlevel,
                "type": mapping_type,
                "factors": factor_str,
                "permutation": permutation_str,
            }
            mapping["mapspace_constraints"].append(entry)

    def add_mapping(mapping, memlevel, entry_str, factor_str, mapping_type="temporal"):
        entry_found = False
        for entry in mapping["mapspace_constraints"]:
            if entry["target"] == memlevel and entry["type"] == mapping_type:
                entry[entry_str] += f" {factor_str}"

    mapping_yaml = f"{config_dir}/{arch_prefix}mapper.yaml"
    mapping_ew_yaml = f"{config_dir}/{arch_prefix}mapper_ew.yaml"

    mapping = utils.parse_yaml(mapping_yaml)
    mapping_ew = utils.parse_yaml(mapping_ew_yaml)


    if not ffmt:
        mapping = utils.parse_yaml(f"{config_dir}/{arch_prefix}mapper_nc.yaml")
        output_subdir = pathlib.Path(
            f"{output_dir}/{arch_prefix}{model_name}/orojenesis_chain_2levels_outputs{batch}{spatial}_opt"
        )
        gen_einsums(
            graph_dir_nc,
            config_dir,
            output_subdir,
            mapping,
            mapping_ew,
            pair_chains_nc,
            layers_dict_nc,
            arch_yaml,
            matmul_only=False,
            num_heads=num_heads,
            force_rerun=force_rerun,
            spatial_factor=spatial_factor
        )
        return

    P_dim = 2 ** (P_prob_log)
    for mapping_postfix in mapping_postfice:

        for p_order in range(0, P_prob_log + 1):
            p_value = 2**p_order

            # Update mapping factors
            permutation_str = "CKPRSQN"
            if mapping_postfix == "_allowk":
                factor_str = f"K=1 R=1 S=1 Q=1 N=1"
                factor_ew_str = f"K=1" if arch_prefix not in ["simba_"] else "C=1"
            elif mapping_postfix == "_allowk_flash":
                factor_str = f"K=1 R=1 S=1 Q=1 N=1"
                factor_ew_str = f"K=1" if arch_prefix not in ["simba_"] else "C=1"
                permutation_str = "PCKRSQN"
            elif mapping_postfix == "_allown":
                factor_str = f"C=1 R=1 S=1 Q=1 N=1"
                factor_ew_str = f"K=1" if arch_prefix not in ["simba_"] else "C=1"
            elif mapping_postfix == "_allown_flash":
                factor_str = f"C=1 R=1 S=1 Q=1 N=1"
                factor_ew_str = f"K=1" if arch_prefix not in ["simba_"] else "C=1"
                permutation_str = "PKCRSQN"
            elif mapping_postfix == "_allowkn":
                factor_str = f"R=1 S=1 Q=1 N=1"
                factor_ew_str = f""  # only when both k and n are allowed we can relax K
                permutation_str = "CKPRSQN"
            elif mapping_postfix == "_allowkn_flash":
                factor_str = f"R=1 S=1 Q=1 N=1"
                factor_ew_str = f""  # only when both k and n are allowed we can relax K
                permutation_str = "CPKRSQN"
            elif mapping_postfix == "_allownk":
                factor_str = f"R=1 S=1 Q=1 N=1"
                factor_ew_str = f""  # only when both k and n are allowed we can relax K
                permutation_str = "KCPRSQN"
            elif mapping_postfix == "_allownk_flash":
                factor_str = f"R=1 S=1 Q=1 N=1"
                factor_ew_str = f""  # only when both k and n are allowed we can relax K
                permutation_str = "KPCRSQN"

            else:
                factor_str = f"K=1 C=1 R=1 S=1 Q=1 N=1"
                factor_ew_str = f"K=1" if arch_prefix not in ["simba_"] else "C=1"

            p_inner_factor = P_dim // p_value
            if arch_prefix == "simba_":
                last_level_mem = "DRAM"
                update_mapping(
                    mapping,
                    last_level_mem,
                    factor_str,
                    permutation_str,
                    mapping_type="temporal",
                )
                update_mapping(
                    mapping_ew,
                    last_level_mem,
                    factor_ew_str,
                    "PC",
                    mapping_type="temporal",
                )

                add_mapping(
                    mapping,
                    "DRAM",
                    "factors",
                    f"P={p_value}",
                    mapping_type="temporal",
                )
            else:
                update_mapping(
                    mapping,
                    "MainMemory",
                    factor_str,
                    permutation_str,
                    mapping_type="temporal",
                )
                update_mapping(
                    mapping_ew,
                    "MainMemory",
                    factor_ew_str,
                    "MK",
                    mapping_type="temporal",
                )

                factor_str = f"P={p_inner_factor} R=1 S=1 Q=1"
                factor_ew_str = f"M={p_inner_factor}"

                update_mapping(
                    mapping,
                    "WeightBuffer",
                    factor_str,
                    "RSPQCKN",
                    mapping_type="temporal",
                )
                update_mapping(
                    mapping_ew,
                    "WeightBuffer",
                    factor_ew_str,
                    "MK",
                    mapping_type="temporal",
                )

            output_subdir = pathlib.Path(
                f"{output_dir}/{arch_prefix}{model_name}/orojenesis_chain_2levels_outputs{batch}{spatial}{mapping_postfix}_p{p_value}"
            )
            gen_einsums(
                graph_dir,
                config_dir,
                output_subdir,
                mapping,
                mapping_ew,
                pair_chains,
                layers_dict,
                arch_yaml,
                matmul_only,
                force_rerun=force_rerun,
            )


if __name__ == "__main__":

    def wrapper(*args, **kwargs):
        print(f"Arguments were: args={args}, kwargs={kwargs}")
        return gen_mappings(*args, **kwargs)

    fire.Fire(gen_mappings)
