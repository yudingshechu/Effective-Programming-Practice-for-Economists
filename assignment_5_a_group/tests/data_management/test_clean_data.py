"""test data management."""
import pandas as pd
import pytest
from assignment_5_a_group.config import TEST_DIR
from assignment_5_a_group.data_management.clean_data import _clean_chs


@pytest.fixture()
def chs():
    """dtaa."""
    chs = pd.read_stata(TEST_DIR / "data_management" / "chs_data.dta")
    return chs


@pytest.fixture()
def bpi():
    """Read data bpi."""
    bpi = pd.read_stata(TEST_DIR / "data_management" / "BEHAVIOR.dta")
    return bpi


@pytest.fixture()
def info():
    """test."""
    info = pd.read_csv(TEST_DIR / "data_management" / "bpi_info.csv")
    return info


@pytest.fixture()
def fulldata():
    """test."""
    ddd = {
        "chs": pd.read_stata(TEST_DIR / "data_management" / "chs_data.dta"),
        "bpi": pd.read_stata(TEST_DIR / "data_management" / "BEHAVIOR.dta"),
        "info": pd.read_csv(TEST_DIR / "data_management" / "bpi_info.csv"),
    }
    return ddd


def test_clean_chs(chs):
    """test."""
    data_clean = _clean_chs(chs)
    assert not {"childid", "momid"}.intersection(set(data_clean.columns))
