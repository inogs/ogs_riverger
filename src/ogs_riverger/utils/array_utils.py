import inspect
import logging

import numpy as np


def get_best_unique_element(
    v: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Given an array `v`` that may contain repetitions of the same elements,
    returns a tuple of two arrays. The first one is a sorted array where every
    element of `v` appears just one, the second one is an array of indices
    `index` such that `v[index]` is equal to the first array we return. An
    index `i` of `index` is selected such that `weights[i] >= weights[j]` for
    every other `j` such that `v[i] == v[j]`.

    This function is similar to `np.unique(return_index=True)` and behaves in
    the same way if `weights == np.arange(len(v))[::-1]` (or any other
    array in descending order).
    """
    logger = logging.getLogger(f"{__name__}.{inspect.stack()[0][3]}")

    # Here we sort the v and the weights accordingly
    sorting_indices = np.argsort(v)
    weights = weights[sorting_indices]
    v = v[sorting_indices]

    # Now we check the position of each element of v; indeed,
    # unique_weights[i] contains the index of the first occurrence of the value
    # unique_v[i] inside the vector `v`; since valid_time is sorted,
    # from `unique_weights[i]` to `unique_weights[i + 1]` there are all the
    # values with a specific value. There we have to choose the one with
    # the maximum weight
    unique_v, unique_weights = np.unique(v, return_index=True)
    logger.debug(
        "There are %s unique values; from %s to %s",
        len(unique_v),
        unique_v[0],
        unique_v[-1],
    )

    # Here we save the indices of the elements that we select
    corresponding_indices = np.empty_like(unique_v, dtype=np.int32)

    # We convert unique_weights to a list because otherwise the static
    # typechecker of Pycharm wrongly reports an error
    unique_weights: list[int] = list(unique_weights)
    for i, i1 in enumerate(unique_weights):
        i2 = len(v) if i == len(unique_weights) - 1 else unique_weights[i + 1]

        logger.debug(
            "There are %s repetition of the element %s",
            i2 - i1,
            unique_v[i],
        )
        # From i1 to i2 we have the same value; now we choose
        # the index of the element with the maximum weight. This means
        # that weight[index_when_sorted] is the maximum weight for the value
        # we are considering. We also have to take into account that this
        # index is valid only after we have sorted the arrays. To retrieve the
        # original ones, we must use the "sorting_indices" array to go back
        # to the original position
        best_value_local_index = int(np.argmax(weights[i1:i2]))
        index_when_sorted = i1 + best_value_local_index

        corresponding_indices[i] = sorting_indices[index_when_sorted]

    return unique_v, corresponding_indices


def find_slice(original: np.ndarray, sliced: np.ndarray) -> tuple[int, int]:
    """Finds the start and end indices of a slice obtained from a 1D array.

    This function takes an original 1D array `v` in ascending or descending
    order and another array `w` that has been obtained by performing a slice
    of `v` (i.e., `w = v[i:j]`). It calculates and returns the two indices
    `i` and `j` that have been used to get the slice `w` from the original
    array `v`.

    Args:
        original (np.ndarray): The original 1D array in ascending or descending
            order.
        sliced (np.ndarray): The sliced array obtained from the original array
            (w = v[i:j]).

    Returns:
        tuple[int, int]: A tuple containing the start index `i` and end
            index `j` used for the slice.

    Raises:
        ValueError: If the sliced array is not a valid slice of the original
            array.

    Example:
        >>> v = np.array([1, 2, 3, 4, 5])
        >>> w = v[1:4]
        >>> _find_slice(v, w)
        (1, 4)
    """
    if len(sliced) > len(original):
        raise ValueError(
            f"The size of the sliced array ({len(sliced)}) is greater than "
            f"the size of the original one ({len(original)})"
        )

    # We consider the first entry of sliced, and we look for the element of
    # `original` that is closest
    i = np.argmin(np.abs(original - sliced[0]))

    # Convert `i` to a standard integer (from a np.int32 or np.int64)
    i = int(i)

    # In this case, the original array ends before we found a corresponding
    # element for each value of the slice
    if i > len(original) - len(sliced):
        raise ValueError(
            f"The sliced array is not a valid slice of the original array. "
            f"The first element of the sliced array is {sliced[0]} but it "
            f"has position {i} of {len(original)} in the original array. "
            f"Since sliced has length {len(sliced)}, it can not be a valid "
            "slice of the original one."
        )

    j = i + sliced.size

    # Check if sliced is a valid slice of original
    if not np.allclose(original[i:j], sliced, atol=0.5e-3):
        raise ValueError(
            "The sliced array is not a valid slice of the original array."
        )

    return i, j
