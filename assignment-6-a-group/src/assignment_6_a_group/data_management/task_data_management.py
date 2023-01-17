import pandas as pd
import pytask
from assignment_6_a_group.config import BLD


@pytask.mark.produces(BLD / "python" / "data" / "raw_data.csv")
def task_clean_data_python(produces):
    url = "https://github.com/LOST-STATS/lost-stats.github.io/raw/source/Model_Estimation/Matching/Data/smoking.csv"
    data = pd.read_csv(url)
    data = data.replace({"Yes": 1, "No": 0, "Female": 1, "Male": 0})
    data.to_csv(produces, index=False)
