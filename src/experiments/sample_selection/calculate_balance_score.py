"""calculate_balance_score.py

Utility function to compute balance scores for dataset pruning.
"""


def calculate_balance_score(
    ds_name: str, remaining_counts: dict[str, int], initial_counts: dict[str, int]
) -> float:
    """
    Computes a score to balance data pruning across multiple heterogeneous datasets.

    This function calculates the ratio of currently retained samples relative to
    the original dataset size. During global deduplication, this score acts as
    a priority metric: if a duplicate is found in two datasets, the dataset
    with the higher balance_score (meaning it has lost less data so far) is
    selected to 'lose' the sample, thereby equalizing the retention rates
    across all processed files.

    Args:
        ds_name (str): The unique identifier or stem name of the dataset.
        remaining_counts (dict[str, int]): A mapping of dataset names to their
        current sample counts in the pipeline.
        initial_counts (dict[str, int]): A mapping of dataset names to their
        starting sample counts before any processing began.

    Returns:
        float: The current retention ratio (0.0 to 1.0). A higher value indicates
            the dataset is more 'saturated' or closer to its original state.
    """
    # 1e-9 safety buffer to prevent division by zero errors
    return remaining_counts[ds_name] / (initial_counts[ds_name] + 1e-9)
