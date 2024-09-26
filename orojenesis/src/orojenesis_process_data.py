import argparse
import logging
import bz2
import re
import pandas as pd
import numpy as np
import os


def construct_argparser():
    """Returns argument parser"""
    parser = argparse.ArgumentParser(description="Run Configuration")
    parser.add_argument(
        "-i",
        "--stats_file",
        type=str,
        default="",
        help="Input Orojenesis Stats File",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        default="",
        help="Output File",
    )
    parser.add_argument(
        "-a",
        "--keep_all_entry",
        action="store_true",
        help="Indicate whether to keep all different buffer sizes and all mappings.",
    )
    parser.add_argument(
        "-e",
        "--keep_one_best_entry",
        action="store_true",
        help="Indicate whether to keep only the best OI and DRAM accesses per buffer size.",
    )

    parser.add_argument(
        "-u",
        "--keep_one_best_entry_across_buf",
        action="store_true",
        help="Indicate whether to keep only the best OI and DRAM accesses accross various buffer sizes.",
    )

    return parser


def post_processing_keep_one_best_entry_across_buf(df):
    df = df.sort_values(by=['bufsize'])
    max_op_int_val = 0
    max_index = []
    for i, row in df.iterrows():
        cur_val = row['OI']
        if cur_val > max_op_int_val:
            max_op_int_val = cur_val
            max_index.append(i)
    df_new = df.loc[max_index]
    return df_new


def process_data(
    stats_file: str,
    output_file: str,
    keep_all_entry: bool = False,
    keep_one_best_entry: bool = False,
    keep_one_best_entry_across_buf: bool = False,
):
    """Process Orojenesis output csv file.

    Args:
    stats_file: The Orojenesis output csv file path from Timeloop run.
    output_file: The output file path to store sorted data.
        keep_all_entry: Indicate whether to keep all different buffer sizes and all mappings.
        keep_one_best_entry: Indicate whether to keep only the best OI and DRAM accesses per buffer size.
        keep_one_best_entry_across_buf: Indicate whether to keep only the best OI and DRAM accesses accross various buffer sizes.
    """

    chunk_size = 10**6

    if keep_all_entry:
        with open(output_file, "w") as f_out:
            # Iterate through the input file in chunks
            for chunk in pd.read_csv(stats_file, chunksize=chunk_size):
                # Perform any necessary data manipulation or analysis on the chunk
                # Append the chunk to the output file
                chunk.to_csv(f_out, header=f_out.tell() == 0, index=False)
        return

    df_final = pd.DataFrame()
    for df in pd.read_csv(stats_file, chunksize=chunk_size, header=None):
        generated_mapping_files = set(df.iloc[:, -1])
        df = df.sort_values(by=[0])

        # Columns are defined as follows:

        # If using log_orojenesis_mappings w/o log_mappings_verbose
        #   0 - Operational intensity
        #   1 - Total buffer size
        #   2 - DRAM accesses
        #   for each tensor t in [0, T) 
        #     3+t*2 - Total tensor buffer occupancy 
        #     3+t*2+1 - Tensor DRAM accesses 
        #   -2 - Compact mapping
        #   -1 - Mapping YAML if log_mappings_yaml is enabled

        # If using log_orojenesis_mappings w/ log_mappings_verbose
        #   0 - Operational intensity
        #   1 - Total buffer size
        #   2 - DRAM accesses
        #   for each buffer level m in [0, M) 
        #     for each tensor t in [0, T) 
        #       3+(m*T+t)*2 - Total tensor occupancy at buffer level m
        #       3+(m*T+t)*2+1 - Tesnor backing-store accesses at buffer level m 
        #   -2 - Compact mapping
        #   -1 - Mapping YAML if log_mappings_yaml is enabled

        new_col_names = list(df.columns) 
        col_names = ['OI', 'bufsize', 'accesses']
        
        for col_idx, col_name in enumerate(col_names):
            new_col_names[col_idx] = col_name

        df.columns = new_col_names

        # group by the bufsize and take the max values of OI and find the idx of max values
        if keep_one_best_entry:
            idx = df.groupby('bufsize')['OI'].idxmax()
            df = df.loc[idx]
        else:  # keep all equally optimal entries
            idx = df.groupby('bufsize')['OI'].transform(max) == df['OI']
            df = df[idx]

        # Only save the mapping that lead to higher op intensity with higher buf utilization
        # Delete the non-optimal mapping yaml files
        if keep_one_best_entry_across_buf:
            df_one_best = post_processing_keep_one_best_entry_across_buf(df)
        else:
            df_one_best = df

        # Delete the non-optimal mapping yaml files
        try:
            optimal_mapping_files = set(df_one_best.iloc[:, -1])
            mapping_files_to_delete = generated_mapping_files - optimal_mapping_files
            for mapping_file in mapping_files_to_delete:
                if os.path.isfile(mapping_file):
                    try:
                        os.remove(mapping_file)
                    except OSError as e:
                        print(
                            f"Failed to remove suboptimal mapping file {mapping_file}!"
                        )
                        print(e.code, e.strerror)
            # Check if all optimal mapping yamls exist
            for mapping_file in optimal_mapping_files:
                if mapping_file != "None" and not os.path.isfile(mapping_file):
                    raise Exception(f"Optimal mapping file {mapping_file} not found!")
        except:
            pass
        df_final = df_final._append(df_one_best, ignore_index=True)

    # need to rerun the following code as we read 1M row at a time
    if keep_one_best_entry:
        idx = df_final.groupby('bufsize')['OI'].idxmax()
        df_final = df_final.loc[idx]
    else:  # keep all equally optimal entries
        idx = df_final.groupby('bufsize')['OI'].transform(max) == df_final['OI']
        df_final = df_final[idx]

    if keep_one_best_entry_across_buf:
        df_final = post_processing_keep_one_best_entry_across_buf(df)


    df_final = df_final.sort_values(by=['bufsize'])
    df_final.to_csv(output_file, header=False, index=False)


if __name__ == "__main__":
    parser = construct_argparser()
    args = parser.parse_args()

    process_data(
        args.stats_file,
        args.output_file,
        args.keep_all_entry,
        args.keep_one_best_entry,
        args.keep_one_best_entry_across_buf,
    )
