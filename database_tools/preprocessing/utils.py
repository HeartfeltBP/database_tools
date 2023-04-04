import os
import typing
import numpy as np
from scipy import signal
from datetime import datetime
from dataclasses import dataclass, InitVar


@dataclass(frozen=True)
class ConfigMapper:

    config: InitVar[dict]

    def __post_init__(self, config: dict):
        for key, value in config.items():
            object.__setattr__(self, key, value)
            

def get_iso_dt() -> None:
    """Get date in ISO string format.

    Returns:
        str: Date in the form YYYYMMDD.
    """
    return datetime.today().isoformat().split('T')[0].replace('-', '')

def build_data_directory(repo_dir: str, partner: str) -> str:
    """Create empty directory for new dataset.

    Args:
        repo_dir (str): Path to database_tools clone.
        partner (str): On of [mimic3, vital].

    Returns:
        data_dir (str): Path to data directory.
    """
    dt = get_iso_dt()
    data_dir = repo_dir + f'{partner}-data-{dt}/'
    os.chdir(repo_dir)
    os.mkdir(data_dir)
    os.mkdir(data_dir + 'data/')
    os.mkdir(data_dir + 'data/lines')
    os.mkdir(data_dir + 'data/records')
    os.mkdir(data_dir + 'data/records/train')
    os.mkdir(data_dir + 'data/records/val')
    os.mkdir(data_dir + 'data/records/test')
    return data_dir

def download(path: str) -> int:
    """
    Download file from url.

    Args:
        path (str): URL of file.

    Returns:
        response (int): Wget response from system.
    """
    response = os.system(f'wget -q -r -np {path}')
    return response

def window(x: np.ndarray, win_len: int, overlap: int) -> typing.List:
    """
    Gets indices for all windows in signal segment.

    Args:
        x (np.ndarray): Signal data.
        win_len (int): Length of windows (including overlap).
        overlap (int): Length of overlap between windows.

    Returns:
        idx (List): List of indices for windows.
    """
    idx = [[0, win_len]]
    idx = idx + [
        [(i * win_len) - overlap,
         ((i + 1) * win_len) - overlap] for i in range(1, int(len(x) / win_len))
    ]
    return idx

def make_equal_len(x: np.ndarray, y: np.ndarray) -> typing.Tuple[np.ndarray]:
    """Make two 1D numpy arrays equal length by padding (with 0s) the
       one that is longer.

    Args:
        x (np.ndarray): 1D array.
        y (np.ndarray): 1D array.

    Returns:
        x, y typing.Tuple[np.ndarray]: Tuple of arrays with one now padded.
    """
    len_x = len(x)
    len_y = len(y)
    if len_x > len_y:
        y = np.pad(y, pad_width=[0, len_x - len_y])
    else:
        x = np.pad(x, pad_width=[0, len_y - len_x])
    return (x, y)

def resample_signal(sig: list, fs_old: int, fs_new: int) -> typing.Tuple[list, int]:
    """Resample a signal to a new sampling rate. This is done with the context
       of a reference length of time in order to produce a result that is
       evenly divisible by the window length (at the new sampling rate).

    Args:
        sig (list): Data.
        fs_old (int): Old sampling rate.
        fs_new (int): New sampling rate.

    Returns:
        resamp (list): Resampled signal.
    """
    frame_len = len(sig)
    frame_time = frame_len / fs_old
    resamp = signal.resample(sig, int(round(frame_time * fs_new, -1)))
    return resamp
