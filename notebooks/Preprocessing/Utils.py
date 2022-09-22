import os
from scipy import signal
from heartpy import peakdetection

def download(path):
    """
    Download file from url.

    Args:
        path (str): URL of file.

    Returns:
        response (int): Wget response from system.
    """
    response = os.system(f'wget -q -r -np {path}')
    return response

def window(x, win_len, overlap):
    """
    Gets indices for all windows in signal segment.

    Args:
        x (np.ndarray): Signal data.
        win_len (int): Length of windows (including overlap).
        overlap (_type_): Length of overlap between windows.

    Returns:
        idx (List): List of indices for windows.
    """
    idx = [[0, win_len]]
    idx = idx + [
        [(i * win_len) - overlap,
         ((i + 1) * win_len) - overlap] for i in range(1, int(len(x) / win_len))
    ]
    return idx

def normalize():
    # Might use sklearn StandardScaler
    return
