import argparse
import pathlib

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description='Draw learning curve from csv file.')
    parser.add_argument(
        '-f',
        '--csv-file',
        help='csv file with training/test results',
        default='results/learning_curve.csv',
        type=pathlib.Path,
    )

    return parser.parse_args()


def cli_main():
    """Command-line interface for drawing the learning curve."""
    args = parse_args()

    acc_df = pd.read_csv(args.csv_file, index_col='train_num_samples')
    acc_df = acc_df.drop(columns='train_ratio')

    # Convert to errors
    err_df = 1 - acc_df
    # Convert to long format for plotting
    err_df = pd.melt(err_df, var_name='set', value_name='error', ignore_index=False)

    ax = sns.lineplot(data=err_df, x='train_num_samples', y='error', hue='set')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('CNN learning curve')

    plt.show()


if __name__ == '__main__':
    cli_main()
