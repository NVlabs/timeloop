import argparse
import logging
import _pickle as cPickle
import bz2
import re
import pandas as pd
import numpy as np
import os


def construct_argparser():
    """ Returns argument parser """
    parser = argparse.ArgumentParser(description='Run Configuration')
    parser.add_argument('-i',
                        '--stats_file',
                        type=str,
                        default='',
                        help='Input OAVES Stats File',
                        )
    parser.add_argument('-o',
                        '--output_file',
                        type=str,
                        default='',
                        help='Output File',
                        )
    parser.add_argument('-a',
                        '--keep_all_entry',
                        action='store_true',
                        help='keep all different buffer sizes and op',
                        )
    parser.add_argument('-e',
                        '--keep_one_best_entry',
                        action='store_true',
                        help='keep all different buffer sizes',
                        )

    parser.add_argument('-u',
                        '--keep_one_best_entry_across_buf',
                        action='store_true',
                        help='keep buffer sizes that achieves the optimal op',
                        )

    return parser


def post_processing_keep_one_best_entry_across_buf(df):
    df = df.sort_values(by=[0])
    max_op_int_val = 0
    max_index = []
    for i, row in df.iterrows():
        cur_val = row[1]
        if cur_val > max_op_int_val:
            max_op_int_val = cur_val
            max_index.append(i)
    df_new = df.loc[max_index]
    return df_new


def process_data(stats_file: str, output_file: str, keep_all_entry: bool = False, keep_one_best_entry: bool = False, keep_one_best_entry_across_buf: bool = False):
    """ Process OAVES output csv file.

    Args:
    stats_file: The OAVES output csv file path from Timeloop run.
    output_file: The output file path to store sorted data.
        keep_all_entry: Indicate whether to keep all different buffer sizes and all mappings.
        keep_one_best_entry: Indicate whether to keep only the best OI and DRAM accesses per buffer size.
        keep_one_best_entry_across_buf: Indicate whether to keep only the best OI and DRAM accesses accross various buffer sizes.
    """

    chunk_size = 10 ** 6

    if keep_all_entry:
        with open(output_file, 'w') as f_out:
            # Iterate through the input file in chunks
            for chunk in pd.read_csv(stats_file, chunksize=chunk_size):
                # Perform any necessary data manipulation or analysis on the chunk
                # Append the chunk to the output file
                chunk.to_csv(f_out, header=f_out.tell()==0, index=False)
        return



    df_final = pd.DataFrame()
    for df in pd.read_csv(stats_file, chunksize=chunk_size, header=None):
        # df = pd.read_csv(stats_file, header=None)
        generated_mapping_files = set(df.iloc[:,4])
        df = df.sort_values(by=[0])

        # Columns are defined as follows:
        # 0-total buffer utilization (B), 1-DRAM operation intensity (op/B), 2-DRAM accesses (word), 3-compact mapping, 4-mapping file path,
        # 5: total operations, 6:6+m*3-per tensor utilization (B), 6+m*3:6+m*3+3-per tensor DRAM accesses (word)

        # group by the buf util index 0 and take the max values of op int at index 1 and find the idx of max values
        # find mapping that matters
        # idx = df.groupby(1)[0].idxmin()
        # df = df.loc[idx]
        if keep_one_best_entry:
            idx = df.groupby(0)[1].idxmax()
            df = df.loc[idx]
        else: # keep all equally optimal entries
            idx = df.groupby(0)[1].transform(max) == df[1]
            df = df[idx]

        # Only save the mapping that lead to higher op intensity with higher buf utilization
        # Delete the non-optimal mapping yaml files
        if keep_one_best_entry_across_buf:
            df_new = post_processing_keep_one_best_entry_across_buf(df)
        else:
            df_new = df

        # Delete the non-optimal mapping yaml files
        try:
            optimal_mapping_files = set(df_new.iloc[:,4])
            mapping_files_to_delete = generated_mapping_files - optimal_mapping_files
            for mapping_file in mapping_files_to_delete:
                if os.path.isfile(mapping_file):
                    try:
                        os.remove(mapping_file)
                    except OSError as e:
                        print (f"Failed to remove suboptimal mapping file {mapping_file}!")
                        print (e.code, e.strerror)
            # Check if all optimal mapping yamls exist
            for mapping_file in optimal_mapping_files:
                if mapping_file != 'None' and not os.path.isfile(mapping_file):
                    raise Exception(f"Optimal mapping file {mapping_file} not found!")
        except:
            pass
        df_final = df_final._append(df_new, ignore_index=True)

    if keep_one_best_entry:
        idx = df_final.groupby(0)[1].idxmax()
        df_final = df_final.loc[idx]
    else: # keep all equally optimal entries
        idx = df_final.groupby(0)[1].transform(max) == df_final[1]
        df_final = df_final[idx]

    if keep_one_best_entry_across_buf:
        df_final = post_processing_keep_one_best_entry_across_buf(df)

    # df_drop = df.drop(df.columns[4], axis=1)
    df_final = df_final.set_index(0)
    df_final.to_csv(output_file, header=False)


if __name__ == "__main__":
    parser = construct_argparser()
    args = parser.parse_args()

    process_data(args.stats_file, args.output_file, args.keep_all_entry, args.keep_one_best_entry, args.keep_one_best_entry_across_buf)
