import pandas as pd
from utils.preprocessing import find_label_segments


def get_num_expressions(results, mapping=None, group=False):
    """
    Return number of repetitions for each expression.
    One repetition is defined as one continuous labeled segment.

    Parameters
    ----------
    results : pd.DataFrame
        DataFrame containing prediction labels in the column 'pred'.
    mapping : dict, optional
        Dictionary mapping integer labels to expression names).
        If provided, labels will be mapped accordingly.
    group : bool, default False
        If True, returns the count for each individual expression type.
        If False, returns the total number of expression segments.

    Returns
    -------
    counts : pd.DataFrame
        A DataFrame containing expression counts.
        - If `group=True`, indexed by expression label
        - If `group=False`, a single-row DataFrame with the total count.
    """

    segments = find_label_segments(results, label_column="pred")
    segments = segments[segments["label"] != 0]
    if mapping is not None:
        segments["label"] = segments["label"].map(mapping)
        labels = list(mapping.values())
    else:
        labels = [1, 2, 3]
    if group:
        counts = segments.groupby("label").size()
        counts = counts.reindex(labels, fill_value=0).reset_index(
            name="expression_count"
        )
        counts = counts.set_index("label")
    else:
        counts = pd.DataFrame({"expression_count": [len(segments)]})
        counts = counts.set_index("expression_count")

    return counts


def normalize_data(
    data, scaler, grouping_columns, data_cols, fit_all_cols=False
):
    """
    Normalize data within groups defined by 'grouping_columns'.

    Parameters:
    - data: pd.DataFrame, the input data.
    - scaler: a Scikit-learn scaler object (e.g., StandardScaler()).
    - grouping_columns: list of columns to group by.
    - data_cols: list of columns to normalize.
    - fit_all_cols: bool, if True, fit scaler on all concatenated columns.

    Returns:
    - df_normalized: pd.DataFrame, normalized data.
    """

    def normalize_group(group, fit_all_cols):
        if fit_all_cols:
            # Concatenate all data columns into a single frame for fitting
            all_cols = pd.concat([group[col] for col in data_cols], axis=1)
            # Fit the scaler on all concatenated columns
            scaler.fit(all_cols)
            # Transform the data columns all at once
            group[data_cols] = scaler.transform(group[data_cols])
        else:
            # Fit and transform each data column individually
            group[data_cols] = scaler.fit_transform(group[data_cols])
        return group

    # Apply normalization to each group
    df_normalized = data.groupby(grouping_columns).apply(
        normalize_group, fit_all_cols
    )
    df_normalized = df_normalized.reset_index(drop=True)

    return df_normalized


def get_grouped_num_expressions(
    results,
    grouping_columns=["Group", "Subject", "Video"],
    mapping=None,
    group_labels=False,
    normalize=False,
    scaler=None,
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
    group_labels : bool, default False
        If True, counts expressions separately by label. If False, returns
        total count per group.
    normalize : bool, default False
        If True, normalize the expression counts using the provided scaler.
    scaler : sklearn.preprocessing object, optional
        A scaler object (e.g., `StandardScaler`) used to normalize
        expression counts. Required if `normalize` is True.

    Returns
    -------
    DataFrame
        A DataFrame with the grouping columns and the number of expressions
        for each expression for each group.
    """
    if normalize:
        if "Video" not in grouping_columns:
            normalization_grouping_cols = grouping_columns + ["Video"]
        print(normalization_grouping_cols)
        grouped = (
            results.groupby(normalization_grouping_cols)
            .apply(
                lambda x: get_num_expressions(
                    x, mapping=mapping, group=group_labels
                )
            )
            .reset_index()
        )
        normalized_data = normalize_data(
            grouped,
            scaler,
            grouping_columns=["Subject"],
            data_cols=["expression_count"],
        )
        if group_labels:
            grouping_columns.append("label")
        grouped = (
            normalized_data.groupby(grouping_columns)["expression_count"]
            .sum()
            .reset_index()
        )
    else:
        grouped = (
            results.groupby(grouping_columns)
            .apply(
                lambda x: get_num_expressions(
                    x, mapping=mapping, group=group_labels
                )
            )
            .reset_index()
        )

    return grouped
