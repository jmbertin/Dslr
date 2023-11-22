import numpy as np


def count(data: np.ndarray) -> int:
    """
    Counts the number of non-NaN elements in a numpy array.
    Args: data (np.ndarray): The array to examine.
    Returns: int: The number of non-NaN elements.
    """
    return len(list(filter(lambda x: not np.isnan(x), data)))

def sum(data: np.ndarray) -> float:
    """
    Calculates the sum of non-NaN elements in a numpy array.
    Args: data (np.ndarray): The array to examine.
    Returns: float: The sum of the non-NaN elements.
    """
    result = 0
    for value in data:
        if not np.isnan(value):
            result += value
    return result

def mean(data: np.ndarray) -> float:
    """
    Calculates the arithmetic mean (average) of non-NaN elements in a numpy array.
    Args: data (np.ndarray): The array to examine.
    Returns: float: The mean of the non-NaN elements.
    Raises: AssertionError: If the data array is empty.
    """
    length = count(data)
    assert length > 0, "Mean can't be calculated on empty data"
    return sum(data) / length

def std(data: np.ndarray) -> float:
    """
    Calculates the standard deviation of non-NaN elements in a numpy array.
    Args: data (np.ndarray): The array to examine.
    Returns: float: The standard deviation of the non-NaN elements.
    Raises: AssertionError: If the data array contains less than two elements.
    """
    length = count(data)
    assert length >= 2, "Std can't be calculated on empty data"
    return np.sqrt(sum((data - mean(data))**2) / (count(data) - 1))

def mini(data: np.ndarray) -> float:
    """
    Finds the smallest non-NaN value in a numpy array.
    Args: data (np.ndarray): The array to examine.
    Returns: float: The smallest non-NaN value.
    """
    minimum: float = np.nan
    for value in data:
        if not np.isnan(value):
            if np.isnan(minimum):
                minimum = value
            elif value < minimum:
                minimum = value
    return minimum

def maxi(data: np.ndarray) -> float:
    """
    Finds the largest non-NaN value in a numpy array.
    Args: data (np.ndarray): The array to examine.
    Returns: float: The largest non-NaN value.
    """
    maximum: float = np.nan
    for value in data:
        if not np.isnan(value):
            if np.isnan(maximum):
                maximum = value
            elif value > maximum:
                maximum = value
    return maximum

def quantile(data: np.ndarray, q: float) -> float:
    """
    Calculates the q-th quantile of the non-NaN elements in a numpy array.
    Args: data (np.ndarray): The array to examine.
          q (float): The quantile to calculate.
    Returns: float: The q-th quantile of the non-NaN elements.
    """
    data = list(filter(lambda x: not np.isnan(x), data))
    return sorted(data)[int(q * count(data))]

def median(data: np.ndarray) -> float:
    """
    Calculates the median of the non-NaN elements in a numpy array.
    Args: data (np.ndarray): The array to examine.
    Returns: float: The median of the non-NaN elements.
    """
    return quantile(data, 0.5)

# Bonus functions

def missing_values(data: np.ndarray) -> int:
    """
    Counts the number of NaN elements in a numpy array.
    Args: data (np.ndarray): The array to examine.
    Returns: int: The number of NaN elements.
    """
    return np.isnan(data).sum()

def variance(data: np.ndarray) -> float:
    """
    Calculates the variance of non-NaN elements in a numpy array.
    Args: data (np.ndarray): The array to examine.
    Returns: float: The variance of the non-NaN elements.
    Raises: AssertionError: If the data array contains less than two elements.
    """
    length = count(data)
    assert length >= 2, "Variance can't be calculated on empty data"
    return sum((data - mean(data))**2) / (count(data) - 1)

def range_value(data: np.ndarray) -> float:
    """
    Calculates the range (difference between max and min) of non-NaN elements in a numpy array.
    Args: data (np.ndarray): The array to examine.
    Returns: float: The range of the non-NaN elements.
    Raises: AssertionError: If the data array is empty.
    """
    assert count(data) > 0, "Range can't be calculated on empty data"
    return maxi(data) - mini(data)
