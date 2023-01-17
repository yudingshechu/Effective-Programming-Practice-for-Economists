"""run monte carlo simulation here, generate data."""
import numpy as np
import pandas as pd


def do_monte_carlo(tp, y_sd, cov_type, meas_sds, nr, seed, n_obs):
    """Implement Monte Carlo simulation.

    Args:
        tp (int): The true coefficients vector of regression model.
        y_sd (float): The standard deviation of error term, i.e. of y.
        cov_type (str): The type of variance-covariance matrix of x,
        input "random" or "deterministic".
        meas_sds (float): The standard deviation of measurement error.
        nr (int): Times of Monte Carlo simulations.
        seed (int): A seed to random number generator.
        n_obs (int): Number of observations.

    Raises:
        ValueError: Illegal cov_type is inputted.

    Returns:
        data: Simulation result for "Bias", "RMSE"
        and "Measurement error standard deviation" for each x variable

    """
    if type(tp) == str:
        raise ValueError("Parameter should not be a string")
    if type(y_sd) == str:
        raise ValueError("y_sd should not be a string")
    if y_sd < 0:
        raise ValueError("y_sd should not be a negative number")
    if type(meas_sds) == str:
        raise ValueError("meas_sds should not be a string")
    if type(nr) == str:
        raise ValueError("nr should not be a string")
    if type(seed) == str:
        raise ValueError("seed should not be a string")
    if type(n_obs) == str:
        raise ValueError("n_obs should not be a string")
    if y_sd < 0:
        raise ValueError("y_sd should not be a negative number")
    assert (meas_sds >= 0).all(), "Only positive allowed for argument."
    assert nr >= 0, "nr can only be >= 0"
    assert n_obs >= 0, "n_obs can only be equal or bigger than zero"

    mean, rng, n_params, names, to_concat = _data_prepare(tp, seed)

    for meas_sd in meas_sds:
        if cov_type == "deterministic":
            cov = np.eye(n_params) + 0.2
        elif cov_type == "random":
            cov = _convariance_random(rng, n_params)
        else:
            raise ValueError(
                f"Invalid cov_type: {cov_type}. Must be 'random' or 'deter.'",
            )

        # Set up a list to which we will append parameter estimates
        estimates = []
        for _ in range(nr):
            x, y = _xy_value(mean, cov, n_obs, y_sd, rng, tp)
            x = _measurement_error(rng, meas_sd, n_obs, x)
            pa = np.matmul(np.matmul(np.linalg.inv(np.matmul(x.T, x)), x.T), y)
            estimates.append(pa)
        df = _results(estimates, tp, names, meas_sd)
        to_concat.append(df)
    # Concatenate the DataFrame
    data = pd.concat(to_concat)

    return data


def _data_prepare(tp, seed):
    """Do a Data prepare.

    Args:
        tp (int): The true coefficients vector of regression model.
        seed (int):  A seed to random number generator.

    Returns:
        mean: Mean vector of multinormal distribution of x.
        rng: Random number generator.
        n_params: number of parameter x.
        names: name of each x variable.
        to_concat: a container which will be used to stor final data.

    """
    mean = np.zeros(len(tp))
    rng = np.random.default_rng(seed)
    n_params = len(tp)
    # Set up parameter names for plotting
    names = [f"x_{i}" for i in range(len(tp))]
    to_concat = []

    return mean, rng, n_params, names, to_concat


def _convariance_random(rng, n_params):
    """Make Cov matrix.

    Args:
        rng (ramdpm_data): a random data generator defined before.
        n_params (int): number of parameters.

    Returns:
        cov: variance-covariance matrix of x generated randomly.

    """
    helper = rng.uniform(low=-1, high=1, size=(n_params, n_params))  # r
    cov = helper @ helper.T + np.eye(n_params)
    return cov


def _xy_value(mean, cov, n_obs, y_sd, rng, tp):
    """Make xy value.

    Args:
        mean (int): Mean vector of multinormal distribution of x.
        cov (float): variance-covariance matrix of x generated randomly.
        n_obs (int):  Number of observations.
        y_sd (float): The standard deviation of error term, i.e. of y.
        rng (ramdpm_data): a random data generator defined before.
        tp (int): The true coefficients vector of regression model.
    Returns:
        x: independent variable
        y: dependent variable

    """
    # Create independent variables
    x = rng.multivariate_normal(mean=mean, cov=cov, size=n_obs)
    # Draw error
    epsilon = rng.normal(loc=0, scale=y_sd, size=n_obs)
    # Calculate y (before adding measurement error!)
    y = x @ tp + epsilon

    return x, y


def _measurement_error(rng, meas_sd, n_obs, x):
    """Generate m.e.

    Args:
        rng (ramdpm_data): a random data generator defined before.
        meas_sd (float):the standard deviation of measurement error.
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


def _results(estimates, tp, names, meas_sd):
    """Generate result data.

    Args:
        estimates (float): Estimated coefficients of regression model.
        tp (int): The true coefficients vector of regression model.
        names: name of each x variable.
        meas_sd (float): the standard deviation of measurement error.

    Returns:
        df: name, bias, RMSE and corresponding Measurement error std.

    """
    import pandas as pd
    import numpy as np

    df = pd.DataFrame()
    deviations = np.array(estimates) - tp
    df["name"] = names
    df["bias"] = deviations.mean(axis=0)
    df["rmse"] = np.sqrt((deviations**2).mean(axis=0))
    df["meas_sd"] = meas_sd

    return df
