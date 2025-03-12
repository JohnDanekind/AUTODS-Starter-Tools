from langchain_core.tools import tool

@tool
def mean(data: list = None):
    """
    Calculate the mean of a dataset.

    Args:
        data: A list of numbers to calculate the mean of.

    Returns:
        The arithmetic mean of the dataset.
    """
    if data is None or len(data) == 0:
        return "Please provide a non-empty list of numbers."
    try:
        numbers = [float(num) for num in data]
        return sum(numbers) / len(numbers)
    except ValueError:
        return "Error: All values must be numeric."


@tool
def stddev(data: list = None):
    """
    Calculate the standard deviation of a dataset.

    Args:
        data: A list of numbers to calculate the standard deviation of.

    Returns:
        The standard deviation of the dataset.
    """
    if data is None or len(data) == 0:
        return "Please provide a non-empty list of numbers."
    try:
        import math
        numbers = [float(num) for num in data]
        avg = sum(numbers) / len(numbers)
        variance = sum([(x - avg) ** 2 for x in numbers]) / len(numbers)
        return math.sqrt(variance)
    except ValueError:
        return "Error: All values must be numeric."


@tool
def sum_data(data: list = None):
    """
    Calculate the sum of a dataset.

    Args:
        data: A list of numbers to sum up.

    Returns:
        The sum of all numbers in the dataset.
    """
    if data is None or len(data) == 0:
        return "Please provide a non-empty list of numbers."
    try:
        numbers = [float(num) for num in data]
        return sum(numbers)
    except ValueError:
        return "Error: All values must be numeric."