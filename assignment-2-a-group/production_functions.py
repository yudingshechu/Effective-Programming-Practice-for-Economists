"""Define production functions."""

from cmath import inf


def general_cobb_douglas(factors, weights, a):
    """Calculate the output of a general Cobb-Douglas production function.

    Args:
        factors (list): Input factors of arbitrary length.
        weights (list): Exponent of the same length.
        a (float): Total factor productivity.

    Returns:
        float: The output of the Cobb-Douglas production function

    """

    output = a
    for i in range(0, len(factors)):
        output = output * factors[i] ** weights[i]

    return output


def general_ces(factors, weights, a, rho):
    """Calculate the output of a general
    Constant Elasticity of Substitution production function.

    Args:
        factors (list): Input factors of arbitrary length.
        weights (list): Exponent of the same length.
        a (float): Total factor productivity.
        rho (float): obtain Leontief function (approaches infinity)
        or Cobb Douglas Function (approaches 0).

    Returns:
        float: The output of the Constant Elasticity of Substitution production function

    """
    s = sum(x * y ** (-rho) for x, y in zip(weights, factors))
    output = a * s ** (-1 / rho)

    return output


def robust_general_ces(factors, weights, a, rho):
    """Calculate the output of a robust general
    Constant Elasticity of Substitution production function.

    Args:
        factors (list): Input factors of arbitrary length.
        weights (list): Exponent of the same length.
        a (float): Total factor productivity.
        rho (float): return approaches to 0 when input is 0.

    Returns:
        float: The output of the Constant Elasticity of Substitution production function

    """

    if rho == 0:
        output = general_cobb_douglas(factors, weights, a)

    else:
        output = general_ces(factors, weights, a, rho)

    return output
