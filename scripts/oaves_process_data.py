import argparse
import logging
import _pickle as cPickle
import bz2
import re
import pandas as pd
import numpy as np


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
                        help='keep all different buffer sizes',
                        )

    return parser

	
def process_data(stats_file: str, output_file: str, keep_all_entry: bool = True):
    """ Process OAVES output csv file.

    Args:
	stats_file: The OAVES output csv file path from Timeloop run.
	output_file: The output file path to store sorted data.
        keep_all_entry: Indicate whether to keep all different buffer sizes.
        
    """	

    df = pd.read_csv(stats_file, header=None)

    # col 0 is buffer size, 1 is op intensity, 2 is dram accesses
    idx = df.groupby(0)[1].idxmax()
    df = df.loc[idx]
    df = df.sort_values(by=[0])
    df = df.set_index(0)

    if keep_all_entry:
        df.to_csv(output_file, header=False)
        return

    max_op_int_val = 0
    max_index = []
    for i, row in df.iterrows():
        cur_val = row[1]
        if cur_val > max_op_int_val:
            max_op_int_val = cur_val
            max_index.append(i)
    
    df_new = df.loc[max_index]
    df_new.to_csv(output_file, header=False)


if __name__ == "__main__":
    parser = construct_argparser()
    args = parser.parse_args()
    
    process_data(args.stats_file, args.output_file, args.keep_all_entry)
