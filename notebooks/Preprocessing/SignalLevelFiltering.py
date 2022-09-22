import numpy as np
from scipy import signal, integrate

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

def get_snr(x, low, high, fs):
    # Estimate spectral power density
    freqs, psd = signal.welch(x, fs)
    freq_res = freqs[1] - freqs[0]

    # Signal power
    idx_sig = np.logical_and(freqs >= low, freqs <= high)
    p_sig = integrate.simps(psd[idx_sig], dx=freq_res)

    # Noise power
    idx_noise_low = freqs < low
    p1 = integrate.simps(psd[idx_noise_low], dx=freq_res)
    idx_noise_high = freqs > high
    p2 = integrate.simps(psd[idx_noise_high], dx=freq_res)
    p_noise = p1 + p2

    # Find SNR and convert to dB
    snr = 10 * np.log10(p_sig / p_noise)
    return snr
