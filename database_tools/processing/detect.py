import itertools
import numpy as np
from neurokit2.ppg.ppg_findpeaks import _ppg_findpeaks_bishop

def detect_flat_lines(x: np.ndarray, n: int) -> bool:
    """
    Flat line detection.

    Args:
        x (np.ndarray): Signal to process.
        n (int): Maximum length of flat line.

    Returns:
        bool: True if n identical values exist consecutively in x.
    """
    return any(sum(1 for _ in g) > (n - 1) for _, g in itertools.groupby(x))

def detect_peaks(sig, show=False, **kwargs):
    """Modified version of neuroki2 ppg_findpeaks method. Returns peaks and troughs
       instead of just peaks. See neurokit2 documentation for original function.
    """
    peaks, troughs = _ppg_findpeaks_bishop(sig, show=show, **kwargs)
    return dict(peaks=peaks[0], troughs=troughs[0])

def detect_notches(sig: np.ndarray, peaks: np.ndarray, troughs: np.ndarray, dx: int = 10, thresh: int = 10) -> list:
    """Detect dichrotic notch by find the maximum velocity
       at least 10 samples after peak and 30 samples before
       the subsequent trough.

    Args:
        sig (np.ndarray): Cardiac signal.
        peaks (list): List of signal peak indices.
        troughs (list): List of signal trough indices.
        dx (int, optional): Spacing between sig values (for np.gradient). Defaults to 10.
        thresh (int, optional): Minimum distance between peak and subsequent notch.

    Returns:
        notches (list): List of dichrotic notch indices.
    """
    # always start with first peak
    try:
        if peaks[0] > troughs[0]:
            troughs = troughs[1::]
    except IndexError:
        return []

    notches = []
    for i, j in zip(peaks, troughs):
        try:
            vel = np.gradient(sig[i:j], dx)
            vel_len = vel.shape[0]
            n = np.argmax(vel[int(vel_len / 100 * 10):int(vel_len / 100 * 90)])
            notches.append(n + i)  # add first index of slice to get correct notch index
        except ValueError:  # gradient fails if slice of sig is too small
            continue

    # look for a notch after the last peak if the highest index is a peak.
    try:
        if peaks[-1] > troughs[-1]:
            try:
                vel = np.gradient(sig[peaks[-1]::], dx)
                vel_len = vel.shape[0]
                n = np.argmax(vel[int(vel_len / 100 * 25):int(vel_len / 100 * 75)])
                notches.append(n + peaks[-1])
            except ValueError:
                pass
    except IndexError:
        pass

    # remove notches that are closer than thresh distance in samples
    a, b = np.meshgrid(notches, peaks)
    notch_to_peak_distances = np.abs(b - a).transpose()
    valid_notch_idx = [i for i, x in enumerate(notch_to_peak_distances) if x[i] >= thresh]
    notches = np.array(notches)[valid_notch_idx]
    return notches
