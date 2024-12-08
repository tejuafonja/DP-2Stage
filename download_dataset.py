import os
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import wget
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-seed", "--seed", default=[1000], nargs="+", type=int)

    parser.add_argument(
        "-name",
        "--dataset-name",
        default="adult",
        choices=["adult", "airline", "heart", "diabetes", "travel"],
        type=str,
    )
    parser.add_argument("-save", "--save-path", default="./data", type=str)

    parser.add_argument("--force", action="store_true")

    parser.add_argument("--split_by_seed", action="store_true")

    parser.add_argument("--train_subset", default=30932, type=int)
    parser.add_argument("--valid_subset", default=1000, type=int)

    parser.add_argument("--use_full_dataset", action="store_true")

    args = parser.parse_args()

    args.data_dir = Path(args.save_path) / args.dataset_name

    return args


def download(baseurl, filenames, filedir):
    for filename in filenames:
        fullurl = urljoin(baseurl, filename)
        print(fullurl)
        filepath = filedir / filename
        print(filepath)

        if not filepath.exists():
            filepath = wget.download(fullurl, out=str(filedir))
            print(f"Downloaded filepath: {filepath}")


def delete(filenames, filedir):
    for filename in filenames:
        filepath = filedir / filename

        if filepath.exists():
            os.remove(filepath)
            print(f"{filepath} has been removed")


def save_files(args, filedir, train, test):
    train_test = pd.concat([train, test])
    train_test = train_test[~train_test.duplicated()].reset_index(drop=True)
    train_test = train_test.sample(len(train_test))
    train_test.to_csv(f"{filedir}/{args.dataset_name}.csv", index=None)
    train.to_csv(f"{filedir}/train.csv", index=None)
    test.to_csv(f"{filedir}/test.csv", index=None)

    demo = train.sample(100, random_state=args.seed[0])
    demo.reset_index(inplace=True, drop=True)
    demo.to_csv(f"{filedir}/demo.csv", index=None)

    print(f"Train/Test/Demo saved in {args.data_dir}")
    print(f"#Train/#Test/#Demo: {len(train)}/{len(test)}/{len(demo)}")


def split_by_seed(args):
    for seed in args.seed:
        filedir = args.data_dir / f"k{seed}"
        os.makedirs(filedir, exist_ok=True)

        if args.use_full_dataset:
            train = pd.read_csv(f"{args.data_dir}/{args.dataset_name}.csv")
        else:
            train = pd.read_csv(f"{args.data_dir}/train.csv")
            test = pd.read_csv(f"{args.data_dir}/test.csv")

        train_subset = train.sample(args.train_subset, random_state=seed)
        train_remain = train[~train.index.isin(train_subset.index)]

        if len(train_remain) > args.valid_subset:
            valid_subset = train_remain.sample(args.valid_subset, random_state=seed)
        else:
            valid_subset = train_remain

        if args.use_full_dataset:
            test = train[
                ~(train.index.isin(train_subset.index))
                & ~(train.index.isin(valid_subset.index))
            ].reset_index(drop=True)

        train_subset = train_subset.reset_index(drop=True)
        valid_subset = valid_subset.reset_index(drop=True)

        train_subset.to_csv(f"{filedir}/train.csv", index=None)
        valid_subset.to_csv(f"{filedir}/valid.csv", index=None)
        test.to_csv(f"{filedir}/test.csv", index=None)

        print(f"Split by Seed: Train/Valid saved in {filedir}")
        print(
            f"#Train/#Valid/#Test: {len(train_subset)}/{len(valid_subset)}/{len(test)}"
        )


def get_download_status(dataset_name, filedir):
    status = []
    # check if data is already downloaded
    for name in [dataset_name, "train", "test", "demo"]:
        if (filedir / f"{name}.csv").exists():
            status.append(True)
        else:
            status.append(False)
        return status


def download_from_kaggle(dataset_identifier, path="."):
    import kaggle

    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(dataset_identifier, path=path, unzip=True)


def download_adult(args):
    if not args.data_dir.exists():
        os.makedirs(args.data_dir)

    baseurl = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/"
    filenames = ["adult.data", "adult.test", "adult.names"]
    filedir = args.data_dir

    status = get_download_status(args.dataset_name, filedir)

    if all(status) and not args.force:
        train = pd.read_csv(args.data_dir / "train.csv")
        test = pd.read_csv(args.data_dir / "test.csv")

        print("Dataset exists. Turn on `--force` to download it again.")
    else:
        # download data
        download(baseurl=baseurl, filenames=filenames, filedir=filedir)

        with open(f"{filedir}/adult.names", "r") as f:
            names = f.read().strip().split("\n")

        col_names = names[96:]
        columns = [i.split(":")[0].lower() for i in col_names] + ["income"]

        # load downloaded data
        train = pd.read_csv(f"{filedir}/adult.data", names=columns)
        test = pd.read_csv(f"{filedir}/adult.test", names=columns, skiprows=1)

        cat_cols = [
            i.split(":")[0]
            for i in col_names
            if i.split(":")[1].strip() != "continuous."
        ] + ["income"]

        # preprocess data.
        for col in train.columns:
            if col in cat_cols:
                train[col] = train[col].apply(lambda x: x.strip().capitalize())

        for col in test.columns:
            if col in cat_cols:
                test[col] = test[col].apply(lambda x: x.strip().capitalize())

        test["income"] = test["income"].apply(lambda x: x.strip("."))
        train["income"] = train["income"].apply(lambda x: x.strip("."))

        # save data as CSV files.
        save_files(args, filedir, train, test)

        # remove the downloaded files
        delete(filenames=filenames, filedir=filedir)

    # train = train.replace("?", "NA")
    # test = test.replace("?", "NA")
    # gpt2 tokenization of ' ?' defaults to '?'
    # which affects deserialization.
    # Easier to preprocess this.

    if args.split_by_seed:
        split_by_seed(args)


def download_airline(args):
    if not args.data_dir.exists():
        os.makedirs(args.data_dir)

    dset_id = "teejmahal20/airline-passenger-satisfaction"
    filenames = ["train.csv", "test.csv"]
    filedir = args.data_dir

    status = get_download_status(args.dataset_name, filedir)

    if all(status) and not args.force:
        train = pd.read_csv(args.data_dir / "train.csv")
        test = pd.read_csv(args.data_dir / "test.csv")

        print("Dataset exists. Turn on `--force` to download it again.")
    else:
        # download data
        download_from_kaggle(dataset_identifier=dset_id, path=filedir)

        # load downloaded data
        train = pd.read_csv(f"{filedir}/train.csv").drop(columns=["Unnamed: 0"])
        test = pd.read_csv(f"{filedir}/test.csv").drop(columns=["Unnamed: 0"])

        # preprocess data.
        for col in train.columns:
            if train[col].dtype == "O":
                train[col] = train[col].apply(
                    lambda x: "-".join(x.strip().capitalize().split())
                )
                test[col] = test[col].apply(
                    lambda x: "-".join(x.strip().capitalize().split())
                )

        train.columns = ["-".join(i.lower().split()) for i in train.columns.tolist()]
        test.columns = ["-".join(i.lower().split()) for i in test.columns.tolist()]

        # save data as CSV files.
        save_files(args, filedir, train, test)

    if args.split_by_seed:
        split_by_seed(args)


def download_travel(args):
    if not args.data_dir.exists():
        raise FileExistsError(f"{args.dataset_name} folder does not exist!")

    if args.split_by_seed:
        split_by_seed(args)


if __name__ == "__main__":
    args = parse_args()

    if args.dataset_name == "adult":
        download_adult(args=args)

    if args.dataset_name == "airline":
        download_airline(args=args)

    if args.dataset_name == "travel":
        download_travel(args=args)
