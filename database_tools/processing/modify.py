import numpy as np
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
