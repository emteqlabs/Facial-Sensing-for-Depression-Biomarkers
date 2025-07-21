import pandas as pd
import numpy as np
from utils.preprocessing import find_label_segments


def get_intensities_per_segments(
    segments, intensity_data, intensity_threshold=0
):
    """Extracts mean intensity from all segments. Each expression segment
    has one value for expression intensity extracted by taking the
    maximum value.

    Parameters
    ----------
    segments : pd.DataFrame
        DataFrame containing segmented expression data with 'start' and 'end'
        columns indicating the segment boundaries (in row indices).
    intensity_data : pd.DataFrame
        DataFrame with an 'Intensity' column containing time-series
        intensity values.
    intensity_threshold : float, default 0
        Minimum intensity value to include a segment in the averaging process.

    Returns
    -------
    float
        The mean intensity across all valid segments. Returns 0 if no
        segment exceedsthe threshold.

    """
    intensities_per_segment = []
    for i, row in segments.iterrows():
        start = int(row["start"])
        end = int(row["end"])
        intensity = intensity_data.iloc[start:end]["Intensity"].max()
        if intensity > intensity_threshold:
            intensities_per_segment.append(intensity)
    if intensities_per_segment:
        mean_intensities = np.mean(intensities_per_segment)
    else:
        mean_intensities = 0

    return mean_intensities


def get_mean_intensities(
    intensity_data,
    mapping=None,
    group_labels=False,
    intensity_threshold=0,
    label_col="Predictions",
):
    """
    Get mean expression intensity per group.

    Parameters
    ----------
    intensity_data : pd.DataFrame
        DataFrame containing time-series intensity data and predicted
        expression labels.
    mapping : dict, optional
        Dictionary mapping numeric expression labels to descriptive names.
        If provided, labels in the results will be mapped accordingly.
    group_labels : bool, default False
        If True, returns mean intensity per expression label.
        If False, returns the overall mean intensity across all segments.
    intensity_threshold : float, default 0
        Minimum intensity value required for a segment to be included in
        the calculation.
    label_col : str, default "Predictions"
        Name of the column in `intensity_data` that contains the expression
        label predictions.

    Returns
    -------
    float or pd.DataFrame
        - If `group_labels` is False: a single float representing the
        overall mean intensity.
        - If `group_labels` is True: a DataFrame with mean intensity
        per expression label.
    """
    if group_labels:
        if mapping is not None:
            all_labels = mapping.values()
        else:
            all_labels = [1, 2, 3]
    segments = find_label_segments(intensity_data, label_column=label_col)
    segments = segments[segments["label"] != 0]
    if segments.empty:
        if group_labels:
            # Return a DataFrame with all labels and intensity 0
            return pd.DataFrame(
                {
                    "label": all_labels,
                    "Expression Intensity": [0] * len(all_labels),
                }
            )
        else:
            # Return just an intensity of 0
            return 0
    if mapping is not None:
        segments["label"] = segments["label"].map(mapping)
    if group_labels:
        mean_intensities = segments.groupby("label").apply(
            lambda x: get_intensities_per_segments(
                x, intensity_data, intensity_threshold
            )
        )
        # Convert the series to a DataFrame and reset index
        mean_intensities = mean_intensities.reset_index()
        mean_intensities.columns = ["label", "Expression Intensity"]

        # Ensure all labels are present, filling missing ones with intensity 0
        mean_intensities = (
            mean_intensities.set_index("label")
            .reindex(all_labels, fill_value=0)
            .reset_index()
        )

    else:
        mean_intensities = get_intensities_per_segments(
            segments, intensity_data
        )
    return mean_intensities


def get_grouped_expression_intensity(
    intensities_data,
    grouping_columns=["Group", "Subject"],
    mapping=None,
    group_labels=False,
    intensity_threshold=0,
    label_col="Predictions",
):
    """
    Compute grouped mean expression intensities across groups.
    For each group (e.g., subject, condition), this function calculates the
    mean intensity of expression segments. Optionally, it can return results
    split by expression label, apply label mappings, and filter out segments
    below a minimum intensity threshold.

    Parameters
    ----------
    intensities_data : pd.DataFrame
        DataFrame containing time-series expression intensity values and
        predicted labels.
    grouping_columns : list of str, optional
        Columns used to group the data before computing mean intensities.
        Default is ["Group", "Subject"].
    mapping : dict, optional
        Dictionary to map numeric expression labels to descriptive strings.
    group_labels : bool, default False
        If True, returns mean intensity per expression label within each group.
        If False, returns the overall mean intensity per group.
    intensity_threshold : float, default 0
        Minimum intensity required for an expression segment to be included.
    label_col : str, default "Predictions"
        Column name in `intensities_data` containing expression predictions.

    Returns
    -------
    pd.DataFrame
        A DataFrame with one row per group and the corresponding expression
        intensities.
        - If `group_labels` is True: includes a column for each
        expression label.
        - If `group_labels` is False: includes a single "Expression Intensity"
        column per group.
    """
    if group_labels:
        grouped_data = intensities_data.groupby(grouping_columns).apply(
            lambda x: get_mean_intensities(
                x,
                mapping,
                group_labels,
                intensity_threshold,
                label_col,
            )
        )
    else:
        grouped_data = intensities_data.groupby(grouping_columns).apply(
            lambda x: pd.Series(
                {
                    "Expression Intensity": get_mean_intensities(
                        x,
                        mapping,
                        group_labels,
                        intensity_threshold,
                        label_col,
                    )
                }
            )
        )

    grouped_data = grouped_data.reset_index()
    cols_to_drop = [col for col in grouped_data.columns if "level" in col]
    if cols_to_drop:
        grouped_data = grouped_data.drop(cols_to_drop, axis=1)

    return grouped_data
