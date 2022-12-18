import numpy as np
import pytest
from monte_carlo import do_monte_carlo
from monte_carlo import _data_prepare
from monte_carlo import _convariance_random_generator
from monte_carlo import _true_xy_value_generator
from monte_carlo import _measurement_error_generator
from monte_carlo import _results_for_each_measurement_error
from sklearn.linear_model import LinearRegression
from numpy.testing import assert_array_equal
from scipy.special import softmax


@pytest.fixture
def setup():
    input = {
        "true_params": np.ones(6),
        "y_sd": 1.5,
        "cov_type": "random",
        "meas_sds": np.linspace(0, 50, 10),
        "n_repetitions": 200,
        "seed": 925408,
        "n_obs": 2000,
    }
    return input


def test_converge(setup):
    data = do_monte_carlo(**setup)
    expect_one = np.ones(8)
    expect = expect_one - 2
    x = data.loc[data["name"] == "x_0", ["bias"]]
    x = x.reset_index()
    actual = x.iloc[1:9, 1]
    assert_array_equal(np.round(actual, decimals=1), np.round(expect, decimals=1))


def test_data_prepare_length(setup):
    assert len(_data_prepare(setup["true_params"], setup["seed"])) == 5


def test_convariance_random_generator_semi_positive_definite(setup):
    rng = np.random.default_rng(setup["seed"])
    cov = _convariance_random_generator(rng, len(setup["true_params"]))
    assert np.all(np.linalg.eigvals(cov) >= 0)


def test_true_xy_value_generator(setup):
    mean, rng, n_params, names, to_concat = _data_prepare(
        setup["true_params"], setup["seed"]
    )
    cov = _convariance_random_generator(rng, len(setup["true_params"]))
    x, y = _true_xy_value_generator(
        mean, cov, setup["n_obs"], setup["y_sd"], rng, setup["true_params"]
    )
    epsilon = rng.normal(loc=0, scale=setup["y_sd"], size=setup["n_obs"])
    assert np.var(y).round(0) == np.var(epsilon).round(0) + np.var(
        x @ setup["true_params"]
    ).round(0)


def test_measurement_error_generator(setup):
    mean, rng, n_params, names, to_concat = _data_prepare(
        setup["true_params"], setup["seed"]
    )
    cov = _convariance_random_generator(rng, len(setup["true_params"]))
    x, y = _true_xy_value_generator(
        mean, cov, setup["n_obs"], setup["y_sd"], rng, setup["true_params"]
    )
    t = _measurement_error_generator(rng, setup["meas_sds"][9], setup["n_obs"], x)
    assert np.std(t[:, 0]).round(-1) == setup["meas_sds"][9]


def test_results_for_each_measurement_error(setup):
    mean, rng, n_params, names, to_concat = _data_prepare(
        setup["true_params"], setup["seed"]
    )
    cov = _convariance_random_generator(rng, len(setup["true_params"]))
    estimates = []
    for _ in range(setup["n_repetitions"]):
        x, y = _true_xy_value_generator(
            mean, cov, setup["n_obs"], setup["y_sd"], rng, setup["true_params"]
        )
        x = _measurement_error_generator(rng, setup["meas_sds"][9], setup["n_obs"], x)
        params = LinearRegression().fit(x, y).coef_
        estimates.append(params)
    df = _results_for_each_measurement_error(
        estimates, setup["true_params"], names, setup["meas_sds"][9]
    )
    assert df["bias"][0].round(0) == 0 - setup["true_params"][0]


# @pytest.mark.xfail
# def test_convariance_random_generator_semi_positive_definite_fail(setup):
#     rng = np.random.default_rng(setup["seed"])
#     cov = _convariance_random_generator(rng, len(setup["true_params"]))
#     cov = cov * -1
#     mean = np.zeros(len(setup["true_params"]))
#     with pytest.raises(ValueError) as excinfo:
#         _true_xy_value_generator(
#             mean, cov, setup["n_obs"], setup["y_sd"], rng, setup["true_params"]
#         )
#     assert excinfo.type == ValueError


# @pytest.mark.xfail
def test_y_sd_fail(setup):
    y_sd = setup["y_sd"] * -1
    with pytest.raises(ValueError) as excinfo:
        do_monte_carlo(
            setup["true_params"],
            y_sd,
            setup["cov_type"],
            setup["meas_sds"],
            setup["n_repetitions"],
            setup["seed"],
            setup["n_obs"],
        )
    assert excinfo.type == ValueError


# @pytest.mark.xfail
def test_measurement_error_fail(setup):
    meas_sds = setup["meas_sds"] * (-1)
    with pytest.raises(AssertionError) as excinfo:
        do_monte_carlo(
            setup["true_params"],
            setup["y_sd"],
            setup["cov_type"],
            meas_sds,
            setup["n_repetitions"],
            setup["seed"],
            setup["n_obs"],
        )
    assert str(excinfo.value) == "Only positive meas_sds allowed for argument."


# @pytest.mark.xfail
def test_input_type_error(setup):
    setup["true_params"] = "here is a string"
    with pytest.raises(ValueError) as excinfo:
        do_monte_carlo(**setup)
    assert str(excinfo.value) == "Parameter should not be a string"
