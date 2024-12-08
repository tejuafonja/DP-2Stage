import argparse
import time
import os
import pandas as pd
from snsynth.pytorch.nn import DPCTGAN, DPGAN
from snsynth.pytorch import PytorchDPSynthesizer
import torch
import random
import numpy as np

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


# Set global reproducibility
def set_global_seed(seed=1000):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


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

    parser.add_argument(
        "--synthesizer",
        type=str,
        default="patectgan",
        help="The name of the synthesizer to use.",
        choices=["dpctgan", "dpgan"],
    )

    parser.add_argument("--max_train_samples", default=None, type=int)

    parser.add_argument("--validation_split_percentage", default=5, type=int)

    parser.add_argument("--efficient_finetuning", default="", type=str)

    parser.add_argument("--output_dir", default="./runs/baselines", type=str)

    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--batch_size", default=32, type=int)

    parser.add_argument("--n_synth_set", default=1, type=int)
    parser.add_argument("--n_synth_samples", default=None, type=int)
    parser.add_argument("--seed", default=1000, type=int)
    parser.add_argument("--target_col", default="income", type=str)

    parser.add_argument("--load_from_checkpoint", default=False, action="store_true")

    parser.add_argument(
        "--target_epsilon", type=float, default=1, help="The privacy budget to spend."
    )
    parser.add_argument(
        "--noise_multiplier",
        type=float,
        default=None,
        help="The scale of noise that will be added to conduct DP-SGD",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1,
        help="The strength of clipping to constrain the contribution of per-sample gradients in DP-SGD",
    )
    parser.add_argument(
        "--target_delta",
        type=float,
        default=1e-5,
        help="The probability of information accidentally being leaked.",
    )

    args = parser.parse_args()
    return args


@timeit
def main():
    args = parse_args()

    # even with all these, still couldn't achieve reproducibility of the model
    set_global_seed(args.seed)

    global OUTPUT_DIR
    OUTPUT_DIR = args.output_dir

    save_path = f"{args.output_dir}/{args.dataset_name}/{args.synthesizer}/eps{args.target_epsilon}-clip{args.max_grad_norm}"

    os.makedirs(save_path, exist_ok=True)

    train_data = pd.read_csv(args.dataset_path)
    nan_column = []

    columns = train_data.columns.to_list()
    discrete_columns = train_data.select_dtypes(include=["object"]).columns

    if not args.load_from_checkpoint:
        if args.synthesizer == "dpctgan":
            model = PytorchDPSynthesizer(
                epsilon=args.target_epsilon,
                gan=DPCTGAN(
                    epochs=args.epochs,
                    epsilon=args.target_epsilon,
                    delta=args.target_delta,
                    max_per_sample_grad_norm=args.max_grad_norm,
                    cuda=True,
                    # disabled_dp=False,
                    # batch_size=args.batch_size,
                    verbose=True,
                ),
                preprocessor=None,
            )

            model.fit(
                train_data,
                # categorical_columns=list(set(columns)),
                categorical_columns=discrete_columns,
                continuous_columns=list(set(columns) - set(discrete_columns)),
                preprocessor_eps=0.1,
                nullable=True,
            )
        elif args.synthesizer == "dpgan":
            model = PytorchDPSynthesizer(
                epsilon=args.target_epsilon,
                gan=DPGAN(
                    epochs=args.epochs,
                    epsilon=args.target_epsilon,
                    # batch_size=args.batch_size,
                    delta=args.target_delta,  # grad norm is 1 by default
                ),
                preprocessor=None,
            )
            model.fit(
                train_data,
                categorical_columns=discrete_columns,
                continuous_columns=list(set(columns) - set(discrete_columns)),
                preprocessor_eps=0.1,
                nullable=True,
            )
        else:
            raise NotImplementedError("sorry!")

        torch.save(
            model,
            os.path.join(save_path, "model.pkl"),
        )
    else:
        model = torch.load(os.path.join(save_path, "model.pkl"))

    print("Training finished")
    n_samples = (
        args.n_synth_samples if args.n_synth_samples is not None else len(train_data)
    )

    metadata = get_metadata(train_data)
    synth_folder = f"{save_path}/synth_data"
    raw_folder = f"{synth_folder}/raw_tables"
    processed_folder = f"{synth_folder}/processed_tables"
    os.makedirs(synth_folder, exist_ok=True)
    os.makedirs(raw_folder, exist_ok=True)
    os.makedirs(processed_folder, exist_ok=True)

    if not args.load_from_checkpoint:
        set_global_seed(args.seed)

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
