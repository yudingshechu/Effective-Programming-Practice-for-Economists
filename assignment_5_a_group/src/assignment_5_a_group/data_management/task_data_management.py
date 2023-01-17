"""Run data management."""
from zipfile import ZipFile

import pandas as pd
import pytask
from assignment_5_a_group.config import BLD
from assignment_5_a_group.config import SRC
from assignment_5_a_group.data_management import clean_data

with ZipFile("src/assignment_5_a_group/data/data.zip", "r") as zip_ref:
    zip_ref.extractall("src/assignment_5_a_group/data")


@pytask.mark.depends_on(
    {
        "data_info": SRC / "data_management" / "data_info.yaml",
        "chs": SRC / "data" / "chs_data.dta",
        "bpi": SRC / "data" / "BEHAVIOR_PROBLEMS_INDEX.dta",
        "info": SRC / "data" / "bpi_variable_info.csv",
    }
)
@pytask.mark.produces(BLD / "python" / "data" / "assignment_3_clean.csv")
def task_clean_data_python(depends_on, produces):
    """Generate cleaned data.

    Args:
        depends_on (_type_): _description_
        produces (_type_): _description_

    """
    chs = pd.read_stata(depends_on["chs"])
    bpi = pd.read_stata(depends_on["bpi"], convert_categoricals=False)
    info = pd.read_csv(depends_on["info"])
    merge2, bpi2 = clean_data(chs, bpi, info)
    merge2.to_csv(produces, index=False)


@pytask.mark.depends_on(
    {
        "data_info": SRC / "data_management" / "data_info.yaml",
        "chs": SRC / "data" / "chs_data.dta",
        "bpi": SRC / "data" / "BEHAVIOR_PROBLEMS_INDEX.dta",
        "info": SRC / "data" / "bpi_variable_info.csv",
    }
)
@pytask.mark.produces(BLD / "python" / "data" / "bpi_clean.csv")
def task_clean_bpi_python(depends_on, produces):
    """Generate cleaned data.

    Args:
        depends_on (_type_): _description_
        produces (_type_): _description_

    """
    chs = pd.read_stata(depends_on["chs"])
    bpi = pd.read_stata(depends_on["bpi"], convert_categoricals=False)
    info = pd.read_csv(depends_on["info"])
    merge2, bpi2 = clean_data(chs, bpi, info)
    bpi2.to_csv(produces, index=False)
