import numpy as np

def load_data_rf(file_path: str):
    """
    Load the dataset from a given file path.
    Args:
        file_path: str, path to the data file
    Returns:
        data: numpy array of shape (n_samples, n_features)
        labels: numpy array of shape (n_samples, 1)
    """
    # Load file except for headers
    whole_csv = np.genfromtxt(file_path, delimiter=",", skip_header=1).astype(float)
    # get labels from 11th column
    labels = whole_csv[:, 10]
    data = np.delete(whole_csv, 10, axis=1)
    return data, labels
