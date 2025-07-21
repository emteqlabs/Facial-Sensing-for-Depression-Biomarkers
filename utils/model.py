import pickle
import pandas as pd
from utils.preprocessing import find_label_segments


def load_model(pkl_file_path):
    """
    Loads a scikit-learn model from a pickle (.pkl) file.

    Parameters:
        pkl_file_path (str): The path to the .pkl file containing the model.

    Returns:
        model: The loaded scikit-learn model.
    """
    try:
        with open(pkl_file_path, "rb") as file:
            model = pickle.load(file)
        print(f"Model loaded successfully from '{pkl_file_path}'.")
        return model
    except FileNotFoundError:
        print(f"Error: File '{pkl_file_path}' not found.")
    except pickle.UnpicklingError:
        print(
            "Error: Failed to unpickle the file. Make sure it's a "
            "valid .pkl file."
        )
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def generate_predictions(
    features,
    model,
    info_columns=[
        "File",
        "Task",
        "Subject",
        "Dataset",
        "Video",
        "Oops",
    ],
):
    """
    Generate facial configurations predictions from extracted features.

    This function separates the metadata from the feature set, applies the
    model to generate predictions and class probabilities, and returns a
    combined DataFrame with predictions, metadata, and per-class probabilities.

    Parameters
    ----------
    features : pd.DataFrame
        DataFrame containing both input features and metadata columns.
    model : sklearn-like classifier
        A trained model with `.predict()` and `.predict_proba()` methods.
    info_columns : list of str, optional
        List of column names in `features` to exclude from the prediction input
        and retain in the result.

    Returns
    -------
    pd.DataFrame
        DataFrame containing:
        - `pred`: predicted class label for each row
        - Original metadata columns from `info_columns`
        - One column per class (e.g., `proba_0`, `proba_1`, ...) with predicted
        probabilities

    """
    test_X = features.drop(info_columns, axis=1)
    pred = model.predict(test_X)
    probas = model.predict_proba(test_X)
    info_results = {col: features[col] for col in info_columns}
    df = pd.DataFrame(
        {
            "pred": pred,
            **info_results,
        }
    )
    for i, class_probas in enumerate(probas.T):
        df[f"proba_{i}"] = class_probas

    return df


def filter_predictions(results, window_threshold=2):
    """
    Filter out short expression predictions based on a minimum duration
    threshold.

    This function removes predicted facial configurations that are shorter
    than or equal to the specified `window_threshold` (in number of windows).
    A segment is considered short if its duration does not exceed the
    threshold, and those predictions are set to 0.

    Parameters
    ----------
    results : pd.DataFrame
        DataFrame containing a 'pred' column with predicted class labels for
        each time window.
    window_threshold : int, default 2
        Minimum number of consecutive windows a prediction must span to
        be retained.
        Segments with duration less than or equal to this threshold will
        be removed.

    Returns
    -------
    pd.DataFrame
        A copy of the input DataFrame with short-duration predictions replaced
        by 0 in the 'pred' column.
    """
    filtered_results = results.copy()
    segments = find_label_segments(results, label_column="pred")
    segments = segments[segments["label"] != 0]
    segments = segments[segments["duration"] <= window_threshold]
    for _, row in segments.iterrows():
        start = int(row["start"])
        end = int(row["end"])
        filtered_results.iloc[start:end, 0] = 0

    return filtered_results
