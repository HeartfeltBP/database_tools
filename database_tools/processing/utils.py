import os
import typing
import numpy as np
from scipy import signal
from datetime import datetime

def get_iso_dt() -> None:
    """Get date in ISO string format.

    Returns:
        str: Date in the form YYYYMMDD.
    """
    return datetime.today().isoformat().split('T')[0].replace('-', '')

def build_data_directory(repo_dir: str, partner: str, date=None) -> str:
    """Create empty directory for new dataset.

    Args:
        repo_dir (str): Path to database_tools clone.
        partner (str): On of [mimic3, vital].

    Returns:
        data_dir (str): Path to data directory.
    """
    if date is None:
        date = get_iso_dt()
    data_dir = repo_dir + f'{partner}-data-{date}/'
    if os.path.exists(data_dir):
        return data_dir
    else:
        os.chdir(repo_dir)
        os.mkdir(data_dir)
        os.mkdir(data_dir + 'data/')
        os.mkdir(data_dir + 'data/lines')
        os.mkdir(data_dir + 'data/records')
        os.mkdir(data_dir + 'data/records/train')
        os.mkdir(data_dir + 'data/records/val')
        os.mkdir(data_dir + 'data/records/test')
        return data_dir

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

def resample_signal(sig: np.ndarray, fs_old: int, fs_new: int) -> typing.Tuple[list, int]:
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
    frame_len = sig.reshape(-1).shape[0]
    frame_time = frame_len / fs_old
    resamp = signal.resample(sig, int(round(frame_time * fs_new, -1)))
    return resamp

def repair_peaks_troughs_idx(peaks: np.ndarray, troughs: np.ndarray) -> typing.Tuple[list, list]:
    """Takes a list of peaks and troughs and removes
       out of order elements. Regardless of which occurs first,
       a peak or a trough, a peak must be followed by a trough
       and vice versa.

    Args:
        peaks (list): Signal peaks.
        troughs (list): Signal troughs.

    Returns:
        first_repaired (list): Input with out of order items removed.
        second_repaired (list): Input with out of order items removed.

        Items are always returned with peaks idx as first tuple item.
    """
    # Configure algorithm to start with lowest index.
    try:
        if peaks[0] < troughs[0]:
            first = peaks
            second = troughs
        else:
            second = peaks
            first = troughs
    except IndexError:
        return (np.array([]), np.array([]))

    first_repaired, second_repaired = [], []  # lists to store outputs
    i_first, i_second = 0, 0  # declare starting indices
    for _ in enumerate(first):
        try:
            poi_1 = first[i_first]
            poi_2 = second[i_second]
            if poi_1 < poi_2:  # first point of interest is before second
                poi_3 = first[i_first + 1]
                if poi_2 < poi_3:  # second point of interest is before third
                    first_repaired.append(poi_1)
                    second_repaired.append(poi_2)
                    i_first += 1
                    i_second += 1
                else:  # first without iterating second
                    i_first += 1
            else: # inverse of other else condition
                i_second += 1
        except IndexError: # catch index error (always thrown in last iteration)
            first_repaired.append(poi_1)
            second_repaired.append(poi_2)

    # remove duplicates
    first_repaired = sorted(list(set(first_repaired)))
    second_repaired = sorted(list(set(second_repaired)))

    # place indices in the correct order
    try:
        if peaks[0] < troughs[0]:
            return (np.array(first_repaired), np.array(second_repaired))
        else:
            return (np.array(second_repaired), np.array(first_repaired))
    except IndexError:
        return (np.array([]), np.array([]))
