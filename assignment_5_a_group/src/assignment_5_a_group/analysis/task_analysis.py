"""generate data."""
import numpy as np
import pytask
from assignment_5_a_group.analysis.predict import do_monte_carlo
from assignment_5_a_group.config import BLD


@pytask.mark.produces(BLD / "python" / "predictions" / "a4.csv")
def task_generate_data(produces):
    """simulate."""
    data = do_monte_carlo(
        np.ones(6), 1.5, "random", np.linspace(0, 5, 10), 200, 925408, 2000
    )
    data.to_csv(produces)
