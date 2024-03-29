import numpy as np
import pandas as pd
import pytest
from assignment_6_a_group.data_management import clean_data
from assignment_6_a_group.config import TEST_DIR
from assignment_6_a_group.utilities import read_yaml


@pytest.fixture()
def data():
    data = pd.read_csv(TEST_DIR / "data_management" / "data_fixture.csv")
    return data


@pytest.fixture()
def data_info():
    data_info = read_yaml(TEST_DIR / "data_management" / "data_info_fixture.yaml")
    return data_info


def test_clean_data_drop_columns(data, data_info):
    data_clean = clean_data(data, data_info)
    assert not set(data_info["columns_to_drop"]).intersection(set(data_clean.columns))


def test_clean_data_dropna(data, data_info):
    data_clean = clean_data(data, data_info)
    assert not data_clean.isnull().any(axis=None)


def test_clean_data_categorical_columns(data, data_info):
    data_clean = clean_data(data, data_info)
    for cat_col in data_info["categorical_columns"]:
        cat_col = data_info["column_rename_mapping"].get(cat_col, cat_col)
        assert data_clean[cat_col].dtype == "category"


def test_clean_data_column_rename(data, data_info):
    data_clean = clean_data(data, data_info)
    old_names = set(data_info["column_rename_mapping"].keys())
    new_names = set(data_info["column_rename_mapping"].values())
    assert not old_names.intersection(set(data_clean.columns))
    assert new_names.intersection(set(data_clean.columns)) == new_names


def test_convert_outcome_to_numerical(data, data_info):
    data_clean = clean_data(data, data_info)
    outcome_name = data_info["outcome"]
    outcome_numerical_name = data_info["outcome_numerical"]
    assert outcome_numerical_name in data_clean.columns
    assert data_clean[outcome_name].dtype == "category"
    assert data_clean[outcome_numerical_name].dtype == np.int8
