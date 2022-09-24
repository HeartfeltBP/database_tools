"""
NOT CURRENTLY IN USE
"""

import numpy as np
from neurokit2.ppg import ppg_findpeaks
from heartpy.preprocessing import flip_signal

def segment_beats(x, fs=125):
    x_flip = flip_signal(x)
    idx = ppg_findpeaks(x_flip, sampling_rate=fs)['PPG_Peaks']
    return idx

def successive_beat_similarity():
    return

def two_signal_beat_similarity():
    return
