import itertools
import numpy as np
from scipy import signal, integrate
from database_tools.processing.utils import make_equal_len

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
    try:
        f_idx = np.sort(np.argpartition(psd[peak_idx], -3)[-3:])
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

def get_beat_similarity(x, troughs, fs, return_beats=False):
    """Calculates beat similarity by segmenting beats at valleys and
       calculating the Pearson correlation coefficient. The final value
       output is the mean of the correlation coefficients calculated from
       every combination (n=2) of beats in the signal.

       Returns -1 if there are less than 2 beats.
       ZeroDivisionError returns -2.

    Args:
        x (np.ndarray): Cardiac signal.
        troughs (list): Trough indices.
        fs (int): Sampling rate of cardiac signal.

    Returns:
        s (float): Mean of beat correlation coeffients.
    """
    # check there are at least 2 peaks
    if (len(troughs) < 2):
        return -1

    neg_len = lambda x : len(x) * -1
    beats = sorted(np.split(x, troughs), key=neg_len)

    aligned_beats = [beats[0]]

    for i, b in enumerate(beats[1::]):
        b_new = np.pad(b, pad_width=[0, len(beats[0]) - len(b)])
        aligned_beats.append(b_new)

    aligned_beats = [b for b in aligned_beats if len(b[b != 0]) > fs / 2]
    if return_beats:
        return aligned_beats

    # Get a list of every combination of beats (ie 3 beats -> [[0, 1], [0, 2], [1, 2]])
    idx = [(i, j) for ((i, _), (j, _)) in itertools.combinations(enumerate([i for i in range(len(aligned_beats))]), 2)]

    s = 0
    for i, j in idx:
        beat1, beat2 = make_equal_len(aligned_beats[i], aligned_beats[j])
        s += get_similarity(beat1, beat2)
    try:
        return s / len(aligned_beats)
    except ZeroDivisionError:
        return -2
