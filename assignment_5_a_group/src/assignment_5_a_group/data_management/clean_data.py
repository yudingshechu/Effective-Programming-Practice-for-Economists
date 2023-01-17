"""data clean functions."""
import numpy as np
import pandas as pd


def clean_data(chs, bpi, info):
    """_summary_.

    Args:
        chs (_type_): _description_
        bpi (_type_): _description_
        info (_type_): _description_

    Returns:
        _type_: _description_

    """
    chs = _clean_chs(chs)
    bpi = _clean_bpi(bpi, info)
    bpi2, merge2 = _merge_dataset(chs, bpi)
    merge2 = _standardization(merge2, bpi2)
    merge2 = _sub_scale_cal(merge2)
    merge2 = merge2.reset_index()
    return merge2, bpi2


def _clean_chs(chs):
    """_summary_.

    Args:
        chs (_type_): _description_

    Returns:
        _type_: _description_

    """
    # load the "chs_data"
    # transfer the column "momid"
    chs["mom_id"] = chs["momid"].astype(int)
    chs = chs.drop(columns="momid")
    chs = chs.drop(chs[(chs["year"] < 1986) | (chs["year"] > 2010)].index)
    chs = chs.drop(chs[(chs["year"] % 2 != 0)].index)
    chs["child_id"] = chs["childid"].astype(int)
    chs = chs.drop(columns="childid")
    chs = chs.set_index(["child_id", "year"])
    return chs


def _clean_bpi(bpi, info):
    """_summary_.

    Args:
        bpi (_type_): _description_
        info (_type_): _description_

    Returns:
        _type_: _description_

    """
    bpi[bpi < 0] = np.nan
    bpi = bpi.drop(columns=bpi.columns.difference(info["nlsy_name"]))
    nlsy_name = info["nlsy_name"]
    r_n = info["readable_name"]
    survey_year = "-" + info["survey_year"]
    for i in range(0, len(nlsy_name)):
        bpi = bpi.rename(columns={nlsy_name[i]: (r_n[i] + survey_year[i])})
    readname = r_n.drop_duplicates()
    bpi = pd.wide_to_long(
        bpi, stubnames=readname, i="childid-invariant", j="year", sep="-"
    )
    bpi = bpi.reset_index()
    bpi["mom_id_bpi"] = bpi["momid-invariant"].astype(int)
    bpi["child_id"] = bpi["childid-invariant"].astype(int)
    bpi["birth_order"] = bpi["birth_order-invariant"]
    bpi = bpi.drop(columns=["childid", "momid"])
    bpi = bpi.drop(columns=["momid-invariant", "childid-invariant"])
    bpi = bpi.drop(columns=["birth_order-invariant"])
    bpi = bpi.set_index(["child_id", "year"])
    for i in range(1, 33):
        if (bpi.iloc[:, i] > 3).sum() == 0:
            continue
        else:
            bpi = bpi.drop(bpi.loc[bpi.iloc[:, i] > 3].index)
    return bpi


def _merge_dataset(chs, bpi):
    """_summary_.

    Args:
        chs (_type_): _description_
        bpi (_type_): _description_

    Returns:
        _type_: _description_

    """
    bpi2 = bpi.iloc[:, 1:33].replace([1, 2, 3], [1, 1, 0])
    bpi2 = pd.concat([bpi2, bpi.iloc[:, [0, 33]]], axis=1)
    merge2 = pd.merge(chs, bpi2, how="left", on=["child_id", "year"])
    return bpi2, merge2


def _standardization(m, bpi2):
    """_summary_.

    Args:
        m (_type_): _description_
        bpi2 (_type_): _description_

    Returns:
        _type_: _description_

    """
    grouped_age = m.groupby("age")
    s_m = grouped_age.aggregate(np.mean)
    standard_var = grouped_age.aggregate(np.var)
    merge_only_len = len(m.columns.difference(bpi2.columns))
    n = -1
    for _ in standard_var.index:
        n = n + 1
        for j in range(merge_only_len, m.shape[1]):
            if standard_var.iloc[n, j - 1] != 0:
                tt = (m.loc[m["age"] == _].iloc[:, j] - s_m.iloc[n, j - 1]) / (
                    standard_var.iloc[n, j - 1]
                ) ** (0.5)
                m.loc[(m["age"] == _), m.columns[j]] = tt.to_frame()
            else:
                continue
    return m


def _sub_scale_cal(merge2):
    """_summary_.

    Args:
        merge2 (_type_): _description_

    Returns:
        _type_: _description_

    """
    cat = ["antisocial", "anxiety", "headstrong", "hyperactive", "peer"]
    for i in cat:
        test = merge2.filter(regex=i, axis=1)

        merge2[i + "_mean_score"] = test.mean(axis=1)
    return merge2
