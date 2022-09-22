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

def bandpass(x, low=0.5, high=8.0, fs=125):
    """
    Filters signal with a 4th order Cheby II filter.

    Args:
        x (np.ndarray): Signal data.
        low (float, optional): Lower frequency in Hz. Defaults to 0.5.
        high (float, optional): Upper frequency in Hz. Defaults to 8.0.
        fs (int, optional): Sampling rate. Defaults to 125.

    Returns:
        x (np.ndarray): Filtered signal.
    """
    # # 4th order butterworth filter
    # btr = signal.butter(
    #     4,
    #     [low, high],
    #     btype='bandpass',
    #     output='sos',
    #     fs=fs
    # )

    cby = signal.cheby2(
        N=4,
        rs=20,
        Wn=[low, high],
        btype='bandpass',
        output='sos',
        fs=fs
    )
    x = signal.sosfiltfilt(cby, x, padtype=None)
    return x

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
