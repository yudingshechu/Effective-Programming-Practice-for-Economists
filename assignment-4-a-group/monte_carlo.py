import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def do_monte_carlo(true_params, y_sd, cov_type, meas_sds, n_repetitions, seed, n_obs):
    """Implement Monte Carlo simulation for a multivariable regression, to show the influence of
    measurement error in the first dependent variable.

    Args:
        true_params (int): The true coefficients vector of regression model.
        y_sd (float): The standard deviation of error term, i.e. of y.
        cov_type (str): The type of variance-covariance matrix of x,
        input "random" or "deterministic".
        meas_sds (float): The standard deviation of measurement error.
        n_repetitions (int): Times of Monte Carlo simulations.
        seed (int): A seed to random number generator.
        n_obs (int): Number of observations.

    Raises:
        ValueError: Illegal cov_type is inputed.

    Returns:
        data: Simulation result for "Bias", "RMSE"
        and "Measurement error standard deviation" for each x variable
    """
    if type(true_params) == str:
        raise ValueError("Parameter should not be a string")
    if type(y_sd) == str:
        raise ValueError("y_sd should not be a string")
    if y_sd < 0:
        raise ValueError("y_sd should not be a negative number")
    if type(meas_sds) == str:
        raise ValueError("meas_sds should not be a string")
    if type(n_repetitions) == str:
        raise ValueError("n_repetitions should not be a string")
    if type(seed) == str:
        raise ValueError("seed should not be a string")
    if type(n_obs) == str:
        raise ValueError("n_obs should not be a string")
    if y_sd < 0:
        raise ValueError("y_sd should not be a negative number")
    assert (meas_sds >= 0).all(), "Only positive meas_sds allowed for argument."
    assert n_repetitions >= 0, "n_repetitions can only be equal or bigger than zero"
    assert n_obs >= 0, "n_obs can only be equal or bigger than zero"

    mean, rng, n_params, names, to_concat = _data_prepare(true_params, seed)

    for meas_sd in meas_sds:
        if cov_type == "deterministic":
            cov = np.eye(n_params) + 0.2
        elif cov_type == "random":
            cov = _convariance_random_generator(rng, n_params)
        else:
            raise ValueError(
                f"Invalid cov_type: {cov_type}. Must be 'random' or 'deterministic.'",
            )

        # Set up a list to which we will append parameter estimates
        estimates = []
        for _ in range(n_repetitions):
            x, y = _true_xy_value_generator(mean, cov, n_obs, y_sd, rng, true_params)
            x = _measurement_error_generator(rng, meas_sd, n_obs, x)
            params = LinearRegression().fit(x, y).coef_
            estimates.append(params)
        df = _results_for_each_measurement_error(estimates, true_params, names, meas_sd)
        to_concat.append(df)
    # Concatenate the DataFrame
    data = pd.concat(to_concat)

    return data


def _data_prepare(true_params, seed):
    """
    Args:
        true_params (int): The true coefficients vector of regression model.
        seed (int):  A seed to random number generator.

    Returns:
        mean: Mean vector of multinormal distribution of x.
        rng: Random number generator.
        n_params: number of parameter x.
        names: name of each x variable.
        to_concat: a container which will be used to stor final data.
    """
    import numpy as np

    mean = np.zeros(len(true_params))
    rng = np.random.default_rng(seed)
    n_params = len(true_params)
    # Set up parameter names for plotting
    names = [f"x_{i}" for i in range(len(true_params))]
    # Initialize list to which we will append DataFrames that are concatenated later
    to_concat = []

    return mean, rng, n_params, names, to_concat


def _convariance_random_generator(rng, n_params):
    """
    Args:
        rng (ramdpm_data_generator): a random data generator defined before.
        n_params (int): number of parameters.

    Returns:
        cov: variance-covariance matrix of x generated randomly.
    """

    import numpy as np

    helper = rng.uniform(low=-1, high=1, size=(n_params, n_params))  # r
    cov = helper @ helper.T + np.eye(n_params)
    return cov


def _true_xy_value_generator(mean, cov, n_obs, y_sd, rng, true_params):
    """
    Args:
        mean (int): Mean vector of multinormal distribution of x.
        cov (float): variance-covariance matrix of x generated randomly.
        n_obs (int):  Number of observations.
        y_sd (float): The standard deviation of error term, i.e. of y.
        rng (ramdpm_data_generator): a random data generator defined before.
        true_params (int): The true coefficients vector of regression model.
    Returns:
        x: independent variable
        y: dependent variable
    """
    # Create independent variables
    x = rng.multivariate_normal(mean=mean, cov=cov, size=n_obs)
    # Draw error
    epsilon = rng.normal(loc=0, scale=y_sd, size=n_obs)
    # Calculate y (before adding measurement error!)
    y = x @ true_params + epsilon

    return x, y


def _measurement_error_generator(rng, meas_sd, n_obs, x):
    """
    Args:
        rng (ramdpm_data_generator): a random data generator defined before.
        meas_sd (float): Iterator in upper loop, the standard deviation of measurement error.
        n_obs (int):  Number of observations.
        x (float): True independent variables.

    Returns:
        x: x with measurement error.
    """
    # Draw measurement error
    meas_error = rng.normal(loc=0, scale=meas_sd, size=n_obs)
    # Add measurement error
    x[:, 0] += meas_error
    return x


def _results_for_each_measurement_error(estimates, true_params, names, meas_sd):
    """
    Args:
        estimates (float): Estimated coefficients of regression model.
        true_params (int): The true coefficients vector of regression model.
        names: name of each x variable.
        meas_sd (float): Iterator in upper loop, the standard deviation of measurement error.

    Returns:
        df: data frame with name, bias, RMSE and corresponding Measurement error std.
    """
    import pandas as pd
    import numpy as np

    df = pd.DataFrame()
    deviations = np.array(estimates) - true_params
    df["name"] = names
    df["bias"] = deviations.mean(axis=0)
    df["rmse"] = np.sqrt((deviations**2).mean(axis=0))
    df["meas_sd"] = meas_sd

    return df
