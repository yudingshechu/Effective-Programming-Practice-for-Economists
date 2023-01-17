import pandas as pd
import pytask
from assignment_6_a_group.analysis.model import calculate_diagnostics
from assignment_6_a_group.analysis.model import model_building_up
from assignment_6_a_group.config import BLD
from assignment_6_a_group.config import SRC
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

kwargs = {"produces": BLD / "python" / "models" / "stats.csv"}

@pytask.mark.depends_on(
    {
        "data": BLD / "python" / "data" / "raw_data.csv",
    }
)
@pytask.mark.task(kwargs=kwargs)
def task_fit_model_python(depends_on, produces):
    datahere = pd.read_csv(depends_on["data"])
    models,x_test_scaled,y_test = model_building_up(datahere)
    stats = pd.DataFrame(
        {
            p: calculate_diagnostics(model = model, x_test=x_test_scaled, y_test=y_test)
            for p, model in models.items()
        }
    ).T.stack()
    stats.name = "value"
    stats.index.names = ["penalty", "outcome"]
    stats = stats.reset_index()
    stats.to_csv(produces, index=False)
