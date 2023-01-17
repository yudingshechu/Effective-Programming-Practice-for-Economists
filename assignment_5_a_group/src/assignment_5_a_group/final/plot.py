"""plot figures for assignment 3 4."""
import numpy as np
import pandas as pd
import plotly.express as px


def plot_regression_by_age(data, a, b):
    """Plot the figure for subscales regression."""
    data = data.set_index(["child_id", "year"])
    rage = data.shape[1] - 5
    ds = data.shape[1]
    sub_scale_bpi_true = data.iloc[:, rage:ds]
    bpi_chs_92 = pd.merge(
        sub_scale_bpi_true,
        data.filter(regex="bpi", axis=1),
        on=["child_id", "year"],
        how="inner",
    )
    bpi_chs_92 = bpi_chs_92.replace(-100, np.nan)
    bpi_chs_92 = bpi_chs_92.dropna()

    colcat = bpi_chs_92.filter(regex=a, axis=1).columns
    fig = px.scatter(
        x=bpi_chs_92.loc[:, colcat[0]], y=bpi_chs_92.loc[:, b], trendline="ols"
    ).update_layout(xaxis_title=a, yaxis_title=b)

    return fig


def heatmap(data, data_2):
    """plot."""
    data = data.set_index(["child_id", "year"])
    merge_only_len = len(data.columns.difference(data_2.columns))
    rag = data.shape[1] - 5
    subpi = data.iloc[:, merge_only_len:rag]
    ddd = subpi.filter(regex=r"(headstrong|antisocial|peer)", axis=1).corr()
    fig = px.imshow(ddd, color_continuous_scale="RdBu_r")
    return fig


def assign4(data):
    """plot."""
    fig = px.line(
        data_frame=data,
        y="bias",
        x="meas_sd",
        color="name",
    )
    return fig
