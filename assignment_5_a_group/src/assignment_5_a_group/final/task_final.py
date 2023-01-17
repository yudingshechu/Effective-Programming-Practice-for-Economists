"""generate figures."""
import pandas as pd
import pytask
from assignment_5_a_group.config import BLD
from assignment_5_a_group.config import bpi_cat
from assignment_5_a_group.config import cat
from assignment_5_a_group.config import SRC
from assignment_5_a_group.final import assign4
from assignment_5_a_group.final import heatmap
from assignment_5_a_group.final import plot_regression_by_age


for a, b in zip(cat, bpi_cat):

    kwargs = {
        "group": a,
        "b": b,
        "produces": BLD / "python" / "figures" / f"subscale_{a}.png",
    }

    @pytask.mark.depends_on(
        {
            "data_info": SRC / "data_management" / "data_info.yaml",
            "data": BLD / "python" / "data" / "assignment_3_clean.csv",
            "data_2": BLD / "python" / "data" / "bpi_clean.csv",
        }
    )
    @pytask.mark.task(id=a, kwargs=kwargs)
    def task_plot_regression_python(depends_on, group, b, produces):
        """Subscale images."""
        data = pd.read_csv(depends_on["data"])
        fig = plot_regression_by_age(data, group, b)
        fig.write_image(produces)


# Heatmap
kwargs = {"produces": BLD / "python" / "figures" / "heatmap.png"}


@pytask.mark.depends_on(
    {
        "data_info": SRC / "data_management" / "data_info.yaml",
        "data": BLD / "python" / "data" / "assignment_3_clean.csv",
        "data_2": BLD / "python" / "data" / "bpi_clean.csv",
    }
)
@pytask.mark.task(kwargs=kwargs)
def task_plot_heatmap_python(depends_on, produces):
    """heatmap."""
    data = pd.read_csv(depends_on["data"])
    data_2 = pd.read_csv(depends_on["data_2"])
    fig = heatmap(data, data_2)
    fig.write_image(produces)


kwargs = {"produces": BLD / "python" / "figures" / "assgn4.png"}


@pytask.mark.depends_on({"data": BLD / "python" / "predictions" / "a4.csv"})
@pytask.mark.task(kwargs=kwargs)
def task_plot_assign4_python(depends_on, produces):
    """Plot for Assignment 4."""
    data = pd.read_csv(depends_on["data"])
    fig = assign4(data)
    fig.write_image(produces)
