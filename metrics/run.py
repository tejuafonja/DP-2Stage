import os
import argparse
import pandas as pd
import numpy as np
from functools import partial
from marginal import histogram_intersection, column_metric_wrapper
from utility import efficacy_test
from exact_duplicates import get_exact_duplicates


def get_parser():
    parser = argparse.ArgumentParser(description="Run model training and evaluation.")

    parser.add_argument(
        "--train_path",
        type=str,
        default="./data/adult/k1000/train.csv",
        help="Path to the training data CSV file",
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default="./data/adult/k1000/test.csv",
        help="Path to the test data CSV file",
    )
    parser.add_argument(
        "--fake_path",
        type=str,
        default="./synth.csv",
        help="Path to the fake data CSV file",
    )
    parser.add_argument(
        "--result_path",
        type=str,
        default="./results_tabular_metrics.csv",
        help="Path to the results CSV file",
    )
    parser.add_argument(
        "--target_name", type=str, default="income", help="Name of the target variable"
    )
    parser.add_argument(
        "--scorer_name",
        type=str,
        choices=["f1", "accuracy", "auc"],
        default="auc",
        help="Scorer name for model evaluation",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        choices=["logistic", "tree", "xgb"],
        default="xgb",
        help="Model name to be used for training",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="classification",
        help="Task type (e.g., classification, regression)",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=1000,
        help="Random state for reproducibility",
    )
    parser.add_argument(
        "--bins", type=int, default=50, help="Number of bins for histograms"
    )
    parser.add_argument(
        "--metric_name",
        type=str,
        choices=[
            "histogram_intersection",
            "efficacy_test",
            "exact_duplicates",
        ],
        default="efficacy_test",
        help="Metric name for evaluation",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main():
    args = get_parser().parse_args()
    train_path = args.train_path
    test_path = args.test_path
    fake_path = args.fake_path
    result_path = args.result_path
    target_name = args.target_name
    scorer_name = args.scorer_name
    model_name = args.model_name
    task = args.task
    random_state = args.random_state
    bins = args.bins
    metric_name = args.metric_name
    overwrite = args.overwrite

    # Load data
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    fake_data = pd.read_csv(fake_path)
    cat_cols = test_data.select_dtypes(exclude=["number"]).columns

    # Load or initialize results DataFrame
    try:
        results_df = pd.read_csv(result_path)

        # Create conditions based on the DataFrame columns
        condition1 = (
            (results_df["fake_path"] == fake_path)
            & (results_df["metric"] == metric_name)
            & (results_df["scorer"] == scorer_name)
            & (results_df.get("model_name", None) == model_name)
            & (results_df.get("target_name", None) == target_name)
            & (results_df.get("fake_size", None) == len(fake_data))
        )

        condition2 = (
            (results_df["fake_path"] == fake_path)
            & (results_df["metric"] == metric_name)
            & (results_df.get("bins", None) == bins)
            & (results_df.get("fake_size", None) == len(fake_data))
        )

        # Combine the conditions
        if not overwrite and (condition1 | condition2).any():
            # If any row matches either condition, raise an error
            matching_row = results_df[condition1 | condition2].iloc[
                0
            ]  # Get the first matching row for the error message
            print(
                matching_row["fake_path"],
                matching_row["metric"],
                matching_row["scorer"],
            )
            print(result_path)
            print(len(result_path))
            raise FileExistsError(
                "Metric has already been computed, use --overwrite to overwrite."
            )

    except FileNotFoundError:
        parent_dir = os.path.dirname(result_path)
        os.makedirs(parent_dir, exist_ok=True)
        column_names = [
            "real_path",
            "fake_path",
            "real_size",
            "fake_size",
            "score",
            "metric",
            "scorer",
            "tag",
        ]
        results_df = pd.DataFrame([], columns=column_names)

    out_dicts = []

    # Histogram Intersection Metric
    if metric_name == "histogram_intersection":
        realdata = test_data
        fakedata = train_data
        tag = "real"

        real_hist = column_metric_wrapper(
            realdata=realdata,
            fakedata=fakedata,
            column_metric=partial(
                histogram_intersection,
                fit_data=pd.concat([realdata, fakedata]),
                bins=bins,
            ),
            cat_cols=cat_cols,
            random_state=random_state,
        ).sort_values(by="column_name")

        out_dicts.append(
            {
                "real_path": test_path,
                "fake_path": train_path,
                "real_size": len(realdata),
                "fake_size": len(fakedata),
                "score": real_hist.score.mean(),
                "metric": metric_name,
                "scorer": "mean",
                "bins": bins,
                "tag": tag,
            }
        )

        realdata = test_data
        fakedata = fake_data
        tag = "fake"
        fake_hist = column_metric_wrapper(
            realdata=realdata,
            fakedata=fakedata,
            column_metric=partial(
                histogram_intersection,
                fit_data=pd.concat([realdata, fakedata]),
                bins=bins,
            ),
            cat_cols=cat_cols,
            random_state=random_state,
        ).sort_values(by="column_name")
        out_dicts.append(
            {
                "real_path": test_path,
                "fake_path": fake_path,
                "real_size": len(realdata),
                "fake_size": len(fakedata),
                "score": fake_hist.score.mean(),
                "metric": metric_name,
                "scorer": "mean",
                "bins": bins,
                "tag": tag,
            }
        )

        l2_dist = np.sqrt(((real_hist.score - fake_hist.score) ** 2).sum())
        out_dicts.append(
            {
                "real_path": test_path,
                "fake_path": fake_path,
                "real_size": len(realdata),
                "fake_size": len(fakedata),
                "score": l2_dist,
                "metric": metric_name,
                "scorer": "l2_dist",
                "bins": bins,
                "tag": tag,
            }
        )

    # Efficacy Test Metric
    if metric_name == "efficacy_test":
        realdata = test_data
        fakedata = train_data
        tag = "real"
        real_effit = efficacy_test(
            realdata=realdata,
            fakedata=fakedata,
            target_name=target_name,
            cat_cols=cat_cols,
            model_name=model_name,
            task=task,
            scorer=scorer_name,
            return_dataframe=False,
            keep_default_size=True,
            transformer=None,
            fit_data=pd.concat([realdata, fakedata]),
            random_state=random_state,
        )

        try:
            class_dist_ = (
                fakedata[target_name].value_counts(normalize=True).round(2).to_dict()
            )
            class_dist = {}
            for k, v in class_dist_.items():
                class_dist[f"{target_name}_{k}"] = v
        except:
            class_dist = {}
        out_dict = {
            "real_path": test_path,
            "fake_path": train_path,
            "real_size": len(realdata),
            "fake_size": len(fakedata),
            "score": real_effit,
            "metric": metric_name,
            "scorer": scorer_name,
            "model_name": model_name,
            "target_name": target_name,
            "tag": tag,
        }
        # out_dict.update(class_dist)

        out_dicts.append(out_dict)

        for k, v in class_dist.items():
            out_dict = {
                "real_path": test_path,
                "fake_path": train_path,
                "real_size": len(realdata),
                "fake_size": len(fakedata),
                "score": v,
                "metric": metric_name,
                "scorer": k,
                "model_name": model_name,
                "target_name": target_name,
                "tag": tag,
            }
            out_dicts.append(out_dict)

        realdata = test_data
        fakedata = fake_data
        tag = "fake"
        fake_effit = efficacy_test(
            realdata=realdata,
            fakedata=fakedata,
            target_name=target_name,
            cat_cols=cat_cols,
            model_name=model_name,
            task=task,
            scorer=scorer_name,
            return_dataframe=False,
            keep_default_size=True,
            transformer=None,
            fit_data=pd.concat([realdata, fakedata]),
            random_state=random_state,
        )

        try:
            class_dist_ = (
                fakedata[target_name].value_counts(normalize=True).round(2).to_dict()
            )
            class_dist = {}
            for k, v in class_dist_.items():
                class_dist[f"{target_name}_{k}"] = v
        except:
            class_dist = {}

        out_dict = {
            "real_path": test_path,
            "fake_path": fake_path,
            "real_size": len(realdata),
            "fake_size": len(fakedata),
            "score": fake_effit,
            "metric": metric_name,
            "scorer": scorer_name,
            "model_name": model_name,
            "target_name": target_name,
            "tag": tag,
        }
        # out_dict.update(class_dist)
        out_dicts.append(out_dict)
        for k, v in class_dist.items():
            out_dict = {
                "real_path": test_path,
                "fake_path": fake_path,
                "real_size": len(realdata),
                "fake_size": len(fakedata),
                "score": v,
                "metric": metric_name,
                "scorer": k,
                "model_name": model_name,
                "target_name": target_name,
                "tag": tag,
            }
            out_dicts.append(out_dict)

    if metric_name == "exact_duplicates":
        realdata = test_data
        fakedata = train_data
        tag = "real"

        real_dupl = get_exact_duplicates(realdata=realdata, fakedata=fakedata)

        out_dicts.append(
            {
                "real_path": test_path,
                "fake_path": train_path,
                "real_size": len(realdata),
                "fake_size": len(fakedata),
                "score": real_dupl,
                "metric": metric_name,
                "scorer": "sum",
                "tag": tag,
            }
        )

        realdata = train_data
        fakedata = fake_data
        tag = "fake_train"
        fake_train_dupl = get_exact_duplicates(realdata=realdata, fakedata=fakedata)
        out_dicts.append(
            {
                "real_path": train_path,
                "fake_path": fake_path,
                "real_size": len(realdata),
                "fake_size": len(fakedata),
                "score": fake_train_dupl,
                "metric": metric_name,
                "scorer": f"sum_{tag}",
                "tag": tag,
            }
        )

        realdata = test_data
        fakedata = fake_data
        tag = "fake"
        fake_test_dupl = get_exact_duplicates(realdata=realdata, fakedata=fakedata)
        out_dicts.append(
            {
                "real_path": test_path,
                "fake_path": fake_path,
                "real_size": len(realdata),
                "fake_size": len(fakedata),
                "score": fake_test_dupl,
                "metric": metric_name,
                "scorer": "sum",
                "tag": tag,
            }
        )

    # Save results
    out_df = pd.DataFrame(out_dicts)
    out_cat = pd.concat([results_df, out_df]).reset_index(drop=True)
    out_cat = out_cat[~out_cat.duplicated()]
    out_cat.to_csv(result_path, index=None)


if __name__ == "__main__":
    main()
