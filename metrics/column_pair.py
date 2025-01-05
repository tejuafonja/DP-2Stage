import pandas as pd
import numpy as np
import dython.nominal as dn
from itertools import combinations


from metrics.utils.util import make_size_equal, column_transformer

__all__ = ["associations_difference", "pairwise_similarity", "correlation_accuracy"]


def associations_difference(
    realdata,
    fakedata,
    cat_cols=None,
    nom_nom_assoc="cramer",
    mean_column_difference=True,
    return_dataframe=False,
    keep_default_size=False,
    random_state=1000,
):
    """Computes the column-pair association matrix difference between `realdata` and `fakedata`.
        Correlation Metrics:
            Numerical-Numerical: `pearson correlation`
            Numerical-Categorical: `correlation ration`
            Categorical-Categorical: `cramer` or `theil`

    Args:
        realdata (pd.DataFrame):
            Realdata to evaluate
        fakedata (pd.DataFrame):
            Fakedata to evaluate
        cat_cols (array-like, optional):
            List of categorical columns. Defaults to None.
        nom_nom_assoc (str, optional):
            Categorical metric to use. Defaults to "cramer".
            Must be one of (`cramer`, `theil`).
        mean_column_difference (bool, optional):
            Whether of not to return correlation difference mean across each column.
            Defaults to `True`.
        return_dataframe (bool, optional):
            Whether or not to return result as pandas dataframe. Defaults to `False`.
        keep_default_size (bool, optional):
            Whether or not to keep default size.
                If `False`, `realdata` and `fakedata` will have equal size.
        random_state (int, optional):
            Set random state number to ensure reproducibility.
            Only applicable if `keep_default_size` is `False`.
            Defaults to `1000`.

    Returns:
        pd.DataFrame or float:
            pd.DataFrame if `mean_column_difference=True`
    """
    __name__ = "associations_difference"

    assert isinstance(realdata, pd.DataFrame)
    assert isinstance(fakedata, pd.DataFrame)

    if not keep_default_size:
        realdata, fakedata = make_size_equal(realdata, fakedata, random_state)

    if cat_cols is None:
        cat_cols = realdata.select_dtypes(exclude=["number"]).columns.to_list()

    real_corr = dn.associations(
        dataset=realdata,
        nominal_columns=cat_cols,
        mark_columns=False,
        nom_nom_assoc=nom_nom_assoc,
        annot=False,
        compute_only=True,
    )["corr"]

    fake_corr = dn.associations(
        dataset=fakedata,
        nominal_columns=cat_cols,
        mark_columns=False,
        nom_nom_assoc=nom_nom_assoc,
        annot=False,
        compute_only=True,
    )["corr"]

    if mean_column_difference:
        result = (
            abs(real_corr - fake_corr)
            .mean()
            .to_frame()
            .reset_index()
            .rename(columns={"index": "column_name", 0: "score"})
        )
        result.loc[:, "metric"] = __name__
        result.loc[:, "normalized_score"] = result["score"]
        result.loc[:, "column_type"] = [
            "categorical" if col in cat_cols else "numerical"
            for col in result["column_name"]
        ]
        return_dataframe = False

    else:
        result = abs(real_corr - fake_corr).mean().mean()

    if return_dataframe:
        result = {"score": result, "normalized_score": result, "metric": __name__}
        result = pd.DataFrame([result])

    return result


def total_variation_distance(p, q):
    """
    Compute Total Variation Distance (TVD) between two probability distributions
    """
    return 1 - (0.5 * np.sum(np.abs(p - q)))


def pairwise_similarity(realdata, fakedata, bins=50, return_dataframe=False):
    """
    Compute the average TVD for all two-way marginals across all attribute pairs.

    Args:
        realdata (pd.DataFrame): DataFrame of the real dataset.
        fakedata (pd.DataFrame): DataFrame of the fake dataset.
        bins (int or array-like, optional):
            Defines the number of equal-width bins in the given range.
            Defaults to 50. If None, the joint is computed for each unique
            value of the continuous columns.
        return_dataframe (bool, optional):
            Whether or not to return result as pandas dataframe. Defaults to `False`.

    Returns:
        float: Average TVD across all two-way marginals.
    """

    __name__ = "pairwise_similarity"

    realdata = realdata.copy()
    fakedata = fakedata.copy()

    # Get all pairwise combinations of columns
    columns = realdata.columns
    attribute_pairs = list(combinations(columns, 2))

    if bins is not None:
        bin_edges = np.linspace(0, 1, bins)
        numerical_cols = realdata.select_dtypes(include=np.number).columns
        print(numerical_cols)
        for col in numerical_cols:
            realdata[col], fakedata[col], _ = column_transformer(
                realdata[col], fakedata[col], kind="minmax"
            )

            realdata[col] = pd.cut(
                realdata[col], bins=bin_edges, labels=False, include_lowest=True
            )
            fakedata[col] = pd.cut(
                fakedata[col], bins=bin_edges, labels=False, include_lowest=True
            ).fillna(0)

    total_tvd = 0
    pair_count = len(attribute_pairs)

    for attr1, attr2 in attribute_pairs:
        # Compute joint distribution for the pair in both dataset
        joint_real = realdata.groupby([attr1, attr2]).size().reset_index(name="count")
        joint_fake = fakedata.groupby([attr1, attr2]).size().reset_index(name="count")

        # Normalize counts to probabilities
        joint_real["prob"] = joint_real["count"] / joint_real["count"].sum()
        joint_fake["prob"] = joint_fake["count"] / joint_fake["count"].sum()

        joint_fake = joint_fake.astype(joint_real.dtypes)
        
        # Align the two distributions
        merged = pd.merge(
            joint_real[[attr1, attr2, "prob"]],
            joint_fake[[attr1, attr2, "prob"]],
            on=[attr1, attr2],
            how="outer",
            suffixes=("_real", "_fake"),
        ).fillna(0)

        # Compute TVD for the pair
        tvd = total_variation_distance(merged["prob_real"], merged["prob_fake"])
        total_tvd += tvd

    # Average acros all pairs
    result = total_tvd / pair_count if pair_count > 0 else 0
    if return_dataframe:
        result = {"score": result, "normalized_score": result, "metric": __name__}
        result = pd.DataFrame([result])

    return result


def assign_levels(v):
    if 0 <= v < 0.1:
        return "low"
    elif 0.1 <= v < 0.3:
        return "weak"
    elif 0.3 <= v < 0.5:
        return "middle"
    else:
        return "strong"


def correlation_accuracy(
    realdata,
    fakedata,
    cat_cols=None,
    nom_nom_assoc="cramer",
    return_dataframe=False,
    keep_default_size=False,
    random_state=1000,
):
    """Computes the column-pair association matrix between `realdata` and `fakedata`,
        assign values based on correlation strength and compute accuracy.
        Correlation Metrics:
            Numerical-Numerical: `pearson correlation`
            Numerical-Categorical: `correlation ration`
            Categorical-Categorical: `cramer` or `theil`

    Args:
        realdata (pd.DataFrame):
            Realdata to evaluate
        fakedata (pd.DataFrame):
            Fakedata to evaluate
        cat_cols (array-like, optional):
            List of categorical columns. Defaults to None.
        nom_nom_assoc (str, optional):
            Categorical metric to use. Defaults to "cramer".
            Must be one of (`cramer`, `theil`).
        return_dataframe (bool, optional):
            Whether or not to return result as pandas dataframe. Defaults to `False`.
        keep_default_size (bool, optional):
            Whether or not to keep default size.
                If `False`, `realdata` and `fakedata` will have equal size.
        random_state (int, optional):
            Set random state number to ensure reproducibility.
            Only applicable if `keep_default_size` is `False`.
            Defaults to `1000`.

    Returns:
        pd.DataFrame or float:
            pd.DataFrame if `mean_column_difference=True`
    """
    __name__ = "correlation_accuracy"

    assert isinstance(realdata, pd.DataFrame)
    assert isinstance(fakedata, pd.DataFrame)

    fakedata = fakedata[realdata.columns]

    if not keep_default_size:
        realdata, fakedata = make_size_equal(realdata, fakedata, random_state)

    if cat_cols is None:
        cat_cols = realdata.select_dtypes(exclude=["number"]).columns.to_list()

    real_corr = dn.associations(
        dataset=realdata,
        nominal_columns=cat_cols,
        mark_columns=False,
        nom_nom_assoc=nom_nom_assoc,
        annot=False,
        compute_only=True,
    )["corr"]

    fake_corr = dn.associations(
        dataset=fakedata,
        nominal_columns=cat_cols,
        mark_columns=False,
        nom_nom_assoc=nom_nom_assoc,
        annot=False,
        compute_only=True,
    )["corr"]

    real_corr_assigned = real_corr.copy()
    for col in real_corr.columns:
        real_corr_assigned[col] = real_corr[col].apply(lambda x: assign_levels(x))

    fake_corr_assigned = fake_corr.copy()
    for col in fake_corr.columns:
        fake_corr_assigned[col] = fake_corr[col].apply(lambda x: assign_levels(x))

    result = (fake_corr_assigned == real_corr_assigned).mean().mean()

    if return_dataframe:
        result = {"score": result, "normalized_score": result, "metric": __name__}
        result = pd.DataFrame([result])

    return result
