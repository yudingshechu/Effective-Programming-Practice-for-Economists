"""Define and evaluate production functions."""
# Define Input values

# Define parameters

# Define Production functions
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

    if rho != 0:
        output = general_ces(factors, weights, a, rho)

    return output


factors = [1, 2, 3]
weights = [0.2, 0.2, 0.6]
a = 1
rho = 0.00000001
rho_0 = 0
test_a = general_cobb_douglas(factors=factors, weights=weights, a=a)
test_b = general_ces(factors=factors, weights=weights, a=a, rho=rho)
test_c = robust_general_ces(factors=factors, weights=weights, a=a, rho=rho_0)
test_d = robust_general_ces(factors=factors, weights=weights, a=a, rho=rho)

print(test_a, test_b, test_c, test_d)

expected_output = 2.22

assert expected_output == round(test_a, 2)

# test_a: result of general Cobb Douglas function
# test_b: result of general CES function given a rho tends to 0
# test_C: result of robust CES function given rho = 0
# the result of test_c supposed to equal to test_b
# test_d: result of robust CES function given rho tends to 0
# the result of test_d supposed to equal to test_a
