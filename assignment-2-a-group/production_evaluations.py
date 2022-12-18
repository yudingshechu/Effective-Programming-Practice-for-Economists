"""Evaluate production functions."""

from production_functions import general_cobb_douglas, general_ces, robust_general_ces

# Define Input values
factors = [1, 2, 3]
weights = [0.2, 0.2, 0.6]


# Define parameters
a = 1
rho = 0.00000001
rho_0 = 0

if __name__ == "__main__":

    # Evaluate production functions.
    test_a = general_cobb_douglas(factors=factors, weights=weights, a=a)
    test_b = general_ces(factors=factors, weights=weights, a=a, rho=rho)
    test_c = robust_general_ces(factors=factors, weights=weights, a=a, rho=rho_0)
    test_d = robust_general_ces(factors=factors, weights=weights, a=a, rho=rho)
    # print the results
print(test_a, test_b, test_c, test_d)

expected_output = 2.22

assert expected_output == round(test_a, 2)

# test_a: result of general Cobb Douglas function
# test_b: result of general CES function given a rho tends to 0
# test_C: result of robust CES function given rho = 0
# the result of test_c supposed to equal to test_b
# test_d: result of robust CES function given rho tends to 0
# the result of test_d supposed to equal to test_a
