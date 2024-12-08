import argparse
import pandas as pd
import numpy as np
from transformers import set_seed
import os

from dataset import (
    get_metadata,
)
from utils import dumpy_config_to_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use.",
        choices=["adult", "airline"],
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./data/adult/train.csv",
        help="The path to the csv dataset file",
    )
    parser.add_argument("--output_dir", default="./data/", type=str)
    parser.add_argument("--output_name", default="train", type=str)

    parser.add_argument("--n_synth_set", default=1, type=int)
    parser.add_argument("--n_synth_samples", default=None, type=int)

    parser.add_argument("--seed", default=1000, type=int)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    dumpy_config_to_json(f"./{args.output_dir}/args.json", vars(args))

    data_df = pd.read_csv(args.dataset_path)

    synth_folder = f"{args.output_dir}"
    os.makedirs(synth_folder, exist_ok=True)
    synth_name = args.output_name

    metadata = get_metadata(data_df)

    for i in range(args.n_synth_set):
        n_samples = (
            args.n_synth_samples if args.n_synth_samples is not None else len(data_df)
        )

        gen_table = generate_uniform_data(metadata, n_samples)

        if args.n_synth_set > 1:
            gen_table.to_csv(f"{synth_folder}/{synth_name}_{i}.csv", index=False)
        else:
            gen_table.to_csv(f"{synth_folder}/{synth_name}.csv", index=False)


def generate_uniform_data(metadata, num_samples):
    # Initialize empty DataFrame
    df = pd.DataFrame()

    # Loop over each column in metadata
    for column, props in metadata.items():
        # print(props['dtype'])

        if props["dtype"].startswith("i"):
            min_value = props["stats"]["min"]
            max_value = props["stats"]["max"]
            # Sample from a Gaussian distribution for numerical columns
            df[column] = np.random.uniform(
                low=min_value, high=max_value + 1, size=num_samples
            ).astype(props["dtype"])
        elif props["dtype"].startswith("o"):
            # Sample from a multinomial distribution for categorical columns
            categories = props["categories"]["unique"]
            df[column] = np.random.choice(categories, num_samples).astype(
                props["dtype"]
            )
        else:
            df[column] = np.random.uniform(
                low=min_value, high=max_value, size=num_samples
            ).astype(props["dtype"])

    return df


def test():
    set_seed(1000)
    metadata = get_metadata(pd.read_csv("./data/adult/demo.csv"))
    df = generate_uniform_data(metadata, num_samples=300)
    print(len(df))
    print(df.head())
    print(df.income.value_counts())
    print(df.age.mean(), df.age.std())
    print(metadata["age"])


if __name__ == "__main__":
    main()
    # test()
