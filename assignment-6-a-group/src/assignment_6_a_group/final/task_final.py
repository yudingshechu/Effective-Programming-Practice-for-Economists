import pandas as pd
import plotly.express as px
import pytask
from assignment_6_a_group.config import BLD
from assignment_6_a_group.config import SRC

kwargs = {"produces": BLD / "python" / "figures" / "model_figures.png"}


@pytask.mark.depends_on(
    {
        "data_info": SRC / "data_management" / "data_info.yaml",
        "stats": BLD / "python" / "models" / "stats.csv",
    }
)
@pytask.mark.task(kwargs=kwargs)
def task_plot_validation_python(depends_on, produces):
    stats = pd.read_csv(depends_on["stats"])
    fig = px.line(
        x="penalty", y="value", facet_col="outcome", data_frame=stats, facet_col_wrap=1
    )
    fig.update_yaxes(matches=None)
    fig.update_layout(width=600, height=1000)
    fig.write_image(produces)
