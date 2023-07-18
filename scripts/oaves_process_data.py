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

    return parser


def process_data(stats_file: str, output_file: str, keep_all_entry: bool = False, keep_one_best_entry: bool = False, keep_one_best_entry_across_buf: bool = False):
    """ Process OAVES output csv file.

    Args:
	stats_file: The OAVES output csv file path from Timeloop run.
	output_file: The output file path to store sorted data.
        keep_all_entry: Indicate whether to keep all different buffer sizes and the corresponding optimal mappings.
        keep_one_best_entry: Indicate whether to keep only the best OI and DRAM accesses per buffer size.
        keep_one_best_entry_across_buf: Indicate whether to keep only the best OI and DRAM accesses accross various buffer sizes.
    """

    df = pd.read_csv(stats_file, header=None)
    generated_mapping_files = set(df.iloc[:,4])

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

    df = df.sort_values(by=[0])

    if keep_all_entry:
        df_new = df

    # only save the mapping that lead to higher op intensity with higher buf utilization
    else:
        if keep_one_best_entry_across_buf:
            max_op_int_val = 0
            max_index = []
            for i, row in df.iterrows():
                cur_val = row[1]
                if cur_val > max_op_int_val:
                    max_op_int_val = cur_val
                    max_index.append(i)
            df_new = df.loc[max_index]
        else:
            df_new = df

        # Delete the non-optimal mapping yaml files
        try:
            optimal_mapping_files = set(df_new.iloc[:,4])
            mapping_files_to_delete = generated_mapping_files - optimal_mapping_files
            for mapping_file in mapping_files_to_delete:
                print(mapping_file)
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

    # df_drop = df.drop(df.columns[4], axis=1)
    df_new = df_new.set_index(0)
    df_new.to_csv(output_file, header=False)


if __name__ == "__main__":
    parser = construct_argparser()
    args = parser.parse_args()

    process_data(args.stats_file, args.output_file, args.keep_all_entry, args.keep_one_best_entry)
