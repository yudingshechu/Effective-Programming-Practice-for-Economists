"""test monte carlo."""
import numpy as np
import pytest
from assignment_5_a_group.analysis.predict import _convariance_random
from assignment_5_a_group.analysis.predict import _data_prepare
from assignment_5_a_group.analysis.predict import _measurement_error
from assignment_5_a_group.analysis.predict import _results
from assignment_5_a_group.analysis.predict import _xy_value
from assignment_5_a_group.analysis.predict import do_monte_carlo
from numpy.testing import assert_array_equal


@pytest.fixture()
def sp():
    """test."""
    ii = {
        "tp": np.ones(6),
        "y_sd": 1.5,
        "cov_type": "random",
        "meas_sds": np.linspace(0, 50, 10),
        "nr": 200,
        "seed": 925408,
        "n_obs": 2000,
    }
    return ii


def test_converge(sp):
    """test."""
    data = do_monte_carlo(**sp)
    expect_one = np.ones(8)
    ex = expect_one - 2
    x = data.loc[data["name"] == "x_0", ["bias"]]
    x = x.reset_index()
    ac = x.iloc[1:9, 1]
    assert_array_equal(np.round(ac, decimals=1), np.round(ex, decimals=1))


def test_data_prepare_length(sp):
    """test."""
    assert len(_data_prepare(sp["tp"], sp["seed"])) == 5


def test_convariance_random_semi_positive_definite(sp):
    """test."""
    rng = np.random.default_rng(sp["seed"])
    cov = _convariance_random(rng, len(sp["tp"]))
    assert np.all(np.linalg.eigvals(cov) >= 0)


def test_true_xy_value(sp):
    """test."""
    mean, rng, n_params, names, to_concat = _data_prepare(sp["tp"], sp["seed"])
    cov = _convariance_random(rng, len(sp["tp"]))
    x, y = _xy_value(mean, cov, sp["n_obs"], sp["y_sd"], rng, sp["tp"])
    ep = rng.normal(loc=0, scale=sp["y_sd"], size=sp["n_obs"])
    assert np.var(y).round(0) == (np.var(ep) + np.var(x @ sp["tp"])).round(0)


def test_measurement_error_generator(sp):
    """test."""
    mean, rng, n_params, names, to_concat = _data_prepare(sp["tp"], sp["seed"])
    cov = _convariance_random(rng, len(sp["tp"]))
    x, y = _xy_value(mean, cov, sp["n_obs"], sp["y_sd"], rng, sp["tp"])
    t = _measurement_error(rng, sp["meas_sds"][9], sp["n_obs"], x)
    assert np.std(t[:, 0]).round(-1) == sp["meas_sds"][9]


def test_results(sp):
    """test."""
    mean, rng, n_params, names, to_concat = _data_prepare(sp["tp"], sp["seed"])
    cov = _convariance_random(rng, len(sp["tp"]))
    estimates = []
    for _ in range(sp["nr"]):
        x, y = _xy_value(mean, cov, sp["n_obs"], sp["y_sd"], rng, sp["tp"])
        x = _measurement_error(rng, sp["meas_sds"][9], sp["n_obs"], x)
        params = np.matmul(np.matmul(np.linalg.inv(np.matmul(x.T, x)), x.T), y)
        estimates.append(params)
    df = _results(estimates, sp["tp"], names, sp["meas_sds"][9])
    assert df["bias"][0].round(0) == 0 - sp["tp"][0]


def test_measurement_error_fail(sp):
    """test."""
    meas_sds = sp["meas_sds"] * (-1)
    with pytest.raises(AssertionError) as excinfo:
        do_monte_carlo(
            sp["tp"],
            sp["y_sd"],
            sp["cov_type"],
            meas_sds,
            sp["nr"],
            sp["seed"],
            sp["n_obs"],
        )
    assert str(excinfo.value) == "Only positive allowed for argument."
