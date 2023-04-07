import numpy as np
from scipy import signal
from database_tools.processing.metrics import get_similarity

def align_signals(ppg, abp, win_len, fs):
    """
    Find the index at which the two signals
    have the largest time correlation.

    Args:
        ppg (np.ndarray): PPG data.
        abp (np.ndarray): ABP data.
        win_len (int): Length of windows.
        fs (int, optional): Sampling rate of signal.

    Returns:
        signals (tuple(np.ndarray)): Aligned PPG and ABP window.
    """
    max_offset = int(fs / 2)

    abp = abp[0:win_len]

    corr = []
    for offset in range(0, max_offset):
        x = ppg[offset : win_len + offset]
        corr.append(get_similarity(x, abp))
    idx = np.argmax(corr)
    ppg_shift = ppg[idx : win_len + idx]
    return (ppg_shift, abp)

def bandpass(x, low, high, fs, method='cheby2'):
    """
    Apply one of the following bandpass filters.
      - 4th order Cheby II filter.
      - 4th order butterworth filter.

    Args:
        x (np.ndarray): Signal data.
        low (float, optional): Lower frequency in Hz.
        high (float, optional): Upper frequency in Hz.
        fs (int, optional): Sampling rate.
        method (str, optional): One of ['cheby2', 'butter']. Defaults to 'cheby2'.

    Returns:
        x (np.ndarray): Filtered signal.
    """
    if method == 'cheby2':
        filt = signal.cheby2(
            N=4,
            rs=20,
            Wn=[low, high],
            btype='bandpass',
            output='sos',
            fs=fs,
        )
    elif method == 'butter':
        filt = signal.butter(
            4,
            [low, high],
            btype='bandpass',
            output='sos',
            fs=fs
        )
    else:
        raise ValueError('Method must be one of [\'cheby2\', \'butter\']')
    x = signal.sosfiltfilt(filt, x, padtype=None)
    return x
