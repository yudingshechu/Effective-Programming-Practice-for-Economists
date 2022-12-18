# Answers


## What do you like and dislike about the starter code in `run.py` (Task 2)
- Like: 1. Uses a lot of existing equations to make the code looks concise.

- Like: 2. The code's logic is fluent and easy to follow.

- Dislike: 1. This code doesn't use function, which makes the code untestable.

- Dislike: 2. Nested loop is hard to read and undersatand at the first glance.

## Is testing that you actually get attenuation bias enough (Task 6)
- Yes, it's enough.

- This method is enough for this measurement error case, but actually it is not showing any trend in our data directly.

- This test choose from the second to the last third estimated coefficients, and test if all of them can be round to -1, this means that all of them should be smaller than -0.5, and this implies that the rate of converage is quite large for this measurement error case.

- This method is not so precise, but it's relatively simple and convenient.

- This test does not use any fixture, which may cause problems in the following tests, but this will be fixed in version 2.0

## Strategies for testing with randomness (Task 9, optional)

- We can test it with some statistics, such as mean and variance of a random variable, and use round() function.

- For a variable which comprises both random and deterministic components, we can decomposite its variance and round them to test.

- In addition, we should remember to always set the same random number generator across our test.
