import pandas as pd


def get_exact_duplicates(realdata, fakedata):
    fakedata = fakedata.astype(realdata.dtypes.to_dict())
    columns = fakedata.columns

    realdata_sorted = realdata.sort_values(by=list(realdata.columns)).reset_index(
        drop=True
    )
    fakedata_sorted = fakedata.sort_values(by=list(fakedata.columns)).reset_index(
        drop=True
    )

    realdata_sorted["tag"] = "real"
    fakedata_sorted["tag"] = "fake"

    realdata_sorted = realdata_sorted.drop_duplicates()
    fakedata_sorted = fakedata_sorted.drop_duplicates()

    df_cat = pd.concat([realdata_sorted, fakedata_sorted], axis=0).reset_index(
        drop=True
    )

    dup = len(
        df_cat[df_cat.duplicated(subset=columns, keep="first")].sort_values(
            by=list(realdata_sorted.columns)
        )
    )

    return dup
