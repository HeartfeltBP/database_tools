import numpy as np

def align_signals(pleth, abp, win_len):
    """
    Find the index at which the two signals
    have the largest time correlation.

    Args:
        pleth (np.ndarray): PLETH data.
        abp (np.ndarray): ABP data.
        win_len (int): Length of windows.

    Returns:
        signals (tuple(np.ndarray)): Aligned PLETH and ABP window.
    """
    fs = 125
    max_offset = int(fs / 2)

    abp = abp[0:win_len]
    
    corr = []
    for offset in range(0, max_offset):
        x = pleth[offset : win_len + offset]
        corr.append(np.sum( x * abp ))
    idx = np.argmax(corr)
    x = pleth[idx : win_len + idx]
    signals = (x, abp)
    return signals

def get_similarity(x, y):
    """
    Calculates time or spectral similarity.

    Args:
        x (np.ndarray): PLETH data.
        y (np.ndarray): ABP data.
        spectral (boolean): If True calculate fft
            of signals.

    Returns:
        coef (float): Pearson correlation coefficient.
    """

    x_bar = np.mean(x)
    y_bar = np.mean(y)
    y_temp = y - y_bar
    x_temp = x - x_bar
    covar = np.sum( (x_temp * y_temp) )
    var = np.sqrt( np.sum( (x_temp ** 2 ) ) * np.sum( (y_temp ** 2) ) )
    coef = covar / var
    return coef

def get_hr():
    return

def get_snr():
    return
