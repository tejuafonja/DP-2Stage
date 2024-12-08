import numpy as np


def laplace_mechanism(value, sensitivity, epsilon):
    noise = np.random.laplace(loc=0.0, scale=sensitivity / epsilon)
    return value + noise


def private_min_max_scaling(data, epsilon):
    sensitivity = (np.max(data) - np.min(data)) / len(data)

    private_min = laplace_mechanism(np.min(data), sensitivity, epsilon / 2)
    private_max = laplace_mechanism(np.max(data), sensitivity, epsilon / 2)

    scaled_data = (data - private_min) / (private_max - private_min)
    return scaled_data


def private_standardization(data, epsilon):
    n = len(data)
    sensitivity_mean = 1 / n
    sensitivity_std = 1 / n  # Approximation for the sensitivity of std

    true_mean = np.mean(data)
    true_std = np.std(data)

    noisy_mean = laplace_mechanism(true_mean, sensitivity_mean, epsilon / 2)
    noisy_std = laplace_mechanism(true_std, sensitivity_std, epsilon / 2)

    standardized_data = (data - noisy_mean) / noisy_std
    return standardized_data


# Example usage
data = np.array([1, 2, 3, 4, 5])
epsilon = 1.0  # Privacy budget
standardized_data = private_standardization(data, epsilon)
print(standardized_data)


# Example usage
data = np.array([1, 2, 3, 4, 5])
epsilon = 1.0  # Privacy budget
scaled_data = private_min_max_scaling(data, epsilon)
print(scaled_data)
