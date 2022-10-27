import warnings
import numpy as np
from scipy import signal, integrate

def bandpass(x, low, high, fs):
    """
    Filters signal with a 4th order Cheby II filter.

    Args:
        x (np.ndarray): Signal data.
        low (float, optional): Lower frequency in Hz.
        high (float, optional): Upper frequency in Hz.
        fs (int, optional): Sampling rate.

    Returns:
        x (np.ndarray): Filtered signal.
    """
    # TODO Test results of butterworth vs cheby2
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
        fs=fs,
    )
    x = signal.sosfiltfilt(cby, x, padtype=None)
    return x

def align_signals(pleth, abp, win_len, fs):
    """
    Find the index at which the two signals
    have the largest time correlation.

    Args:
        pleth (np.ndarray): PLETH data.
        abp (np.ndarray): ABP data.
        win_len (int): Length of windows.
        fs (int, optional): Sampling rate of signal.

    Returns:
        signals (tuple(np.ndarray)): Aligned PLETH and ABP window.
    """
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
        x (np.ndarray): PPG data.
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
    with np.errstate(divide='ignore', invalid='ignore'):
        try:
            coef = covar / var
        except (ZeroDivisionError, RuntimeWarning):
            coef = 0
    return coef

def get_snr(x, low, high, df, fs):
    """
    Calculate the Signal-to-noise ratio (SNR) of the cardiac signal.
    Density of spectrum between low and high frequencies is considered
    signal power. Density of spectrum outside low to high frequency
    band is considered signal noise. F0 is estimated to be the frequency
    at which the power spectrum is at its maximum.

    Args:
        x (np.ndarray): Cardiac signal data.
        low (float): Lower frequency in Hz.
        high (float): Upper frequency in Hz.
        df (float): Delta f for calculating power.
        fs (int): Sampling rate of signal.

    Returns:
        snr (float): SNR of signal in dB.
        f0 (float): Fundamental frequency of signal in Hz.
    """
    # Estimate spectral power density
    freqs, psd = signal.periodogram(x, fs, nfft=2048)
    freq_res = freqs[1] - freqs[0]

    dpdf = np.diff(psd) / np.diff(freqs)
    peak_idx = np.where(np.diff(np.sign(dpdf)))[0] + 1
    peak_x = freqs[peak_idx]
    f_idx = np.sort(np.argpartition(psd[peak_idx], -3)[-3:])
    try:
        f0, f1, f2 = peak_x[f_idx]
    except:
        return -10, 0

    # Signal power
    idx_sig_fund = np.logical_and(freqs >= f0 - df, freqs <= f0 + df)
    idx_sig_harm1 = np.logical_and(freqs >= f1 - df, freqs <= f1 + df)
    idx_sig_harm2 = np.logical_and(freqs >= f2 - df, freqs <= f2 + df)

    # Try statements are easiest way of preventing errors during integration
    try:
        p_sig_fund = integrate.simps(psd[idx_sig_fund], dx=freq_res)
    except:
        p_sig_fund = 0
    try:
        p_sig_harm1 = integrate.simps(psd[idx_sig_harm1], dx=freq_res)
    except:
        p_sig_harm1 = 0
    try:
        p_sig_harm2 = integrate.simps(psd[idx_sig_harm2], dx=freq_res)
    except:
        p_sig_harm2 = 0
    p_sig = p_sig_fund + p_sig_harm1 + p_sig_harm2

    # Noise power
    idx_cardiac = np.logical_and(freqs >= low, freqs <= high)

    try:
        p_cardiac = integrate.simps(psd[idx_cardiac], dx=freq_res)
    except:
        p_cardiac = 0
    p_noise = p_cardiac - p_sig

    # Try, except to prevent divide by 0 error
    with np.errstate(divide='ignore', invalid='ignore'):
        try:
            # Find SNR and convert to dB
            snr = 10 * np.log10(p_sig / p_noise)
        except (ZeroDivisionError, RuntimeWarning):
            snr = -10
    return snr, f0

def flat_lines(x):
    """
    Flat line detection.

    Args:
        x (np.ndarray): Signal to process.

    Returns:
        bool: True if flat lines are present and False otherwise.
    """
    for i, val in enumerate(x):
        if (i + 2) >= len(x):
            break
        if (x[i + 1] == val) & (x[i + 2] == val):
            return True
    return False
