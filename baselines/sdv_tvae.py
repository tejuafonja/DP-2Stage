from ctgan import TVAE
import argparse
import pandas as pd
import time
from transformers import set_seed
import os

from utils.dataset import postprocess_data, get_metadata


def timeit(func):
    def wrapper():
        start_time = time.time()
        func()
        end_time = time.time()
        time_elapsed = end_time - start_time
        with open(f"{OUTPUT_DIR}/elapsed_time.txt", "a") as f:
            f.write(
                f"Time elapsed: {time_elapsed} secs / {time_elapsed/60} mins / {time_elapsed/3600} hrs\n"
            )

    return wrapper


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
        default="./data/adult/k1000/train.csv",
        help="The path to the csv dataset file",
    )
    parser.add_argument("--output_dir", default="./runs/tvae", type=str)

    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--device", default="cuda", type=str)

    parser.add_argument("--do_train", default=False, action="store_true")
    parser.add_argument("--do_generate", default=False, action="store_true")
    parser.add_argument("--load_from_dir", default=True, action="store_true")

    parser.add_argument("--n_synth_set", default=1, type=int)
    parser.add_argument("--n_synth_samples", default=None, type=int)
    parser.add_argument("--seed", default=1000, type=int)

    args = parser.parse_args()
    return args


@timeit
def main():
    args = parse_args()
    set_seed(args.seed)

    global OUTPUT_DIR
    OUTPUT_DIR = args.output_dir

    args.batch_size = None
    synthesizer = "tvae"
    save_path = (
        f"{args.output_dir}/{args.dataset_name}/{synthesizer}/bs{args.batch_size}"
    )

    os.makedirs(save_path, exist_ok=True)

    OUTPUT_DIR = save_path

    train_data = pd.read_csv(args.dataset_path)
    nan_column = []
    for i in train_data:
        if train_data[i].isna().any():
            nan_column.append(i)
            train_data[i] = train_data[i].fillna(-1)  # fill na

    model = TVAE(
        epochs=args.epochs, cuda=True if args.device == "cuda" else False, verbose=True
    )

    # Names of the columns that are discrete
    discrete_columns = train_data.select_dtypes(include=["object"]).columns

    if args.do_train:
        model.fit(train_data, discrete_columns)
        model.save(os.path.join(args.output_dir, "model.pt"))

    print("Training finished")
    if args.do_generate:
        if not args.do_train and args.load_from_dir:
            model = model.load(os.path.join(save_path, "model.pt"))

        metadata = get_metadata(train_data)
        synth_folder = f"{save_path}/synth_data"
        synth_folder = f"{save_path}/synth_data"
        raw_folder = f"{synth_folder}/raw_tables"
        processed_folder = f"{synth_folder}/processed_tables"
        os.makedirs(synth_folder, exist_ok=True)
        os.makedirs(raw_folder, exist_ok=True)
        os.makedirs(processed_folder, exist_ok=True)

        n_samples = (
            args.n_synth_samples
            if args.n_synth_samples is not None
            else len(train_data)
        )

        for i in range(args.n_synth_set):
            processed_tables, raw_tables = sample(
                n_samples, model, metadata=metadata, nan_column=nan_column
            )
            print(i)
            raw_tables.to_csv(f"{raw_folder}/synth_{i}.csv", index=False)
            processed_tables.to_csv(f"{processed_folder}/synth_{i}.csv", index=False)


def sample(n_samples, model, metadata=None, nan_column=[]):
    raw_tables = []
    processed_tables = []

    remaining = 0

    while remaining < n_samples:
        raw_table = model.sample(n_samples)
        raw_tables.append(raw_table)

        if len(nan_column) != 0:
            for i in nan_column:
                raw_table[i] = raw_table[i].apply(lambda x: None if x == -1 else x)

        processed_table = postprocess_data(raw_table, metadata=metadata, dropna=True)

        processed_tables.append(processed_table)
        remaining += len(processed_table)

    raw_tables = pd.concat(raw_tables).reset_index(drop=True)
    processed_tables = pd.concat(processed_tables).reset_index(drop=True)
    processed_tables = processed_tables.head(n_samples)

    return processed_tables, raw_tables


if __name__ == "__main__":
    main()
