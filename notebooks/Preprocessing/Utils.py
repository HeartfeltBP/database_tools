import os
from random import sample
from scipy import signal
from heartpy import peakdetection

def download(path):
    response = os.system(f'wget -q -r -np {path}')
    if response == 0:
        return True
    return False

def bandpass(x, low=0.5, high=8.0, fs=125):
    # 4th order butterworth filter
    btr = signal.butter(
        4,
        [low, high],
        btype='bandpass',
        output='sos',
        fs=fs
    )
    x = signal.sosfiltfilt(btr, x, padtype=None)
    return x

def window(x, win_len, overlap, fs=125):
    win = peakdetection.make_windows(
        data=x,
        sample_rate=fs,
        windowsize=win_len,
        overlap=overlap,
        min_size=win_len,
    )
    return win

def normalize():
    # Might use sklearn StandardScaler
    return
