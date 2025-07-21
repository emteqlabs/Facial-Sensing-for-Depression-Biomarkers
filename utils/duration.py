import pandas as pd
from utils.preprocessing import find_label_segments, windows_to_seconds


def get_expression_duration(results, mapping=None, average=False, group=True):
    """
    Return number of repetitions for each expression.
    One repetition is defined as one continuous labeled segment

    Parameters
    ----------
    results : pd.DataFrame
        DataFrame containing predicted expression labels in the column 'pred'.
    mapping : dict, optional
        Dictionary mapping integer labels to descriptive expression names.
        If provided, the output will use these mapped labels.
    average : bool, default False
        If True, returns the average duration of expressions; otherwise,
        returns the duration of each detected segment.
    group : bool, default True
        If True, results are grouped by expression label.
        If False, returns either a single average value or a flat list
        of durations.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing expression durations.
        - If `average=True` and `group=True`: a DataFrame with average duration
        per expression.
        - If `average=True` and `group=False`: a single-row DataFrame with
        overall average duration.
        - If `average=False` and `group=True`: durations for individual
        segments with expression labels.
        - If `average=False` and `group=False`: a flat list of all segment
        durations.
    """
    segments = find_label_segments(results, label_column="pred")
    segments = segments[segments["label"] != 0]
    if group:
        if mapping is not None:
            all_labels = mapping.values()
        else:
            all_labels = [1, 2, 3]
    if not segments.empty:
        if average:
            if mapping is not None:
                segments["label"] = segments["label"].map(mapping)
            if group:
                durations = (
                    segments.groupby("label")["duration"]
                    .mean()
                    .apply(windows_to_seconds)
                )
                result = pd.DataFrame(durations).reset_index()
                # If label not found set value to 0
                result = (
                    result.set_index("label")
                    .reindex(all_labels, fill_value=0)
                    .reset_index()
                )

            else:
                durations = (
                    segments["duration"].apply(windows_to_seconds).mean()
                )
                result = pd.DataFrame([[durations]], columns=["duration"])
        else:
            durations = (
                segments["duration"]
                .apply(windows_to_seconds)
                .reset_index(drop=True)
            )
            if mapping is not None:
                segments["label"] = segments["label"].map(mapping)
            labels = segments["label"].reset_index(drop=True)
            result = pd.DataFrame({"duration": durations, "label": labels})
        return result
    else:
        if group:
            data = {"duration": [0] * len(all_labels), "label": all_labels}
            return pd.DataFrame(data)
        else:
            return pd.DataFrame({"duration": [0]})


def get_grouped_expressions_duration(
    results,
    grouping_columns=["Group", "Subject", "Video"],
    mapping=None,
    average=False,
    group=True,
):
    """
    Compute the number of expressions for each expression for each group and
    return the results as a DataFrame.

    Parameters
    ----------
    results : DataFrame
        The input DataFrame containing the data to be analyzed.
    grouping_columns : list of str, optional
        A list of columns to group by. Default is ["Group", "Subject", "Video"].
    mapping : dict, optional
        A dictionary to map the original labels to new labels in the resultant
        DataFrame
    Returns
    -------
    pd.DataFrame
        A DataFrame with the grouping columns and the number of expressions
        for each expression for each group.
    """
    grouped = (
        results.groupby(grouping_columns)
        .apply(
            lambda df: get_expression_duration(
                df, mapping=mapping, average=average, group=group
            )
        )
        .reset_index(level=-1, drop=True)
        .reset_index()
    )

    return grouped
