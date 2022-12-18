import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly
from pathlib import Path
from monte_carlo import do_monte_carlo

if __name__ == "__main__":
    data = do_monte_carlo(
        np.ones(6), 1.5, "random", np.linspace(0, 5, 10), 200, 925408, 2000
    )

    fig = px.line(
        data_frame=data,
        y="bias",
        x="meas_sd",
        color="name",
    )
    fig.show()
    # ==================================================================================
    # Save data and plot
    # ==================================================================================
    BLD = Path("bld")
    if not BLD.exists():
        BLD.mkdir()
    data.to_pickle(BLD / "results.pkl")
    fig.write_image(BLD / "bias.png")
