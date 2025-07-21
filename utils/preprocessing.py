import pandas as pd
import numpy as np


def windows_to_seconds(n_windows, n_win_size=25, n_win_slide=5):
    """Converts duration of N number of windows to seconds"""
    n_duration = n_win_size + n_win_slide * (n_windows - 1)
    t_duration = n_duration / 50
    return t_duration


def find_subsegments_indices(values):
    """
    A function used to find uninterrupted segments in a series of values and
    returns an array where each row contains the start and end index of samples
    belonging to one uninterrupted segment.

    Parameters
    ----------
    values : numpy.array
        A 1-dimensional numpy array of values.

    Returns
    -------
    segments : numpy.ndarray
        A 2-dimensional numpy array where each row contains the start and end
        index of samples belonging to one uninterrupted segment.

    """

    # If input is pd.Series transform it to numpy array
    if isinstance(values, pd.Series):
        values = values.values
    if len(values) == 0:
        raise ValueError("Segments cannot be extracted from an empty list")

    changes = np.flatnonzero(values[:-1] != values[1:]) + 1
    segments = np.column_stack(
        (
            np.concatenate(([0], changes)),
            np.concatenate((changes, [len(values)])),
        )
    )
    return segments


def get_segment_label(data, segments, label_column):
    """
    A function used to find uninterrupted segments in a series of values and
    returns an array where each row contains the start and end index of samples
    belonging to one uninterrupted segment.

    Parameters
    ----------
    data : pandas.DataFrame
        A dataframe from which the label will be extracted,
        based on the provided segments
    segments : numpy.ndarray
        A 2-dimensional numpy array where each row contains the start and end
        index of samples belonging to one uninterrupted segment.
    label_column : str
        The name of the label column

    Returns
    -------
    labels : list
        A list of labels corresponding to the number of segments given

    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data type is not appropriate.")
    try:
        iter(segments)
    except TypeError:
        raise ValueError('Provided values for "segments" is not iterable')
    if label_column not in data.columns:
        raise ValueError("Target column not found in DataFrame")

    labels = []
    for segment in segments:
        sel_label = data.iloc[
            segment[0] : segment[1], data.columns.get_loc(label_column)
        ]
        if sel_label.shape[0] != 0:
            labels.append(sel_label.unique()[0])
    return labels


def find_label_segments(data, label_column):
    """
    A function used to find uninterrupted segments in a dataframe and returns
    a new dataframe where each row contains the start index, end index,
    and the label belonging to one uninterrupted segment.

    Parameters
    ----------
    data : pandas.DataFrame
        A dataframe from which the segments will be extracted
    label_column : str
        The name of the column from which subsegments will be created

    Returns
    -------
    res : pandas.DataFrame
        A pandas dataframe where each row contains the start index, end index,
        and the label belonging to one uninterrupted segment.

    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input type is not appropriate.")
    if label_column not in data.columns:
        raise ValueError("Target column not found in DataFrame")

    segments = find_subsegments_indices(data[label_column])
    labels = get_segment_label(data, segments, label_column)
    if len(labels) != len(segments):
        raise ValueError(
            f"Length mismatch between segments {(len(segments))}"
            + f" and labels {(len(labels))}"
        )
    segments = pd.DataFrame(segments, columns=["start", "end"])
    segments["duration"] = segments["end"] - segments["start"]
    segments = segments.astype("int")
    segments["label"] = labels
    return segments
