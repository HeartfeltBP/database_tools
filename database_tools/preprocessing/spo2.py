import math
import os
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import pandas as pd
from neurokit2.ppg.ppg_findpeaks import _ppg_findpeaks_bishop

def find_peaks(ppg_cleaned, show=False, **kwargs):
    """Modified version of neuroki2 ppg_findpeaks method. Returns peaks and troughs
       instead of just peaks. See neurokit2 documentation for original function.
    """
    peaks, troughs = _ppg_findpeaks_bishop(ppg_cleaned, show=show, **kwargs)
    return {'peaks': peaks[0], 'troughs': troughs[0]}

def calc_spo2(ppg_r, ppg_i, return_peaks=False):
    try:
        res = find_peaks(ppg_i)
        ir_peaks, ir_troughs = res['peaks'], res['troughs']

        res = find_peaks(ppg_r)
        red_peaks, red_troughs = res['peaks'], res['troughs']

        red_high, red_low = np.max(ppg_r[red_peaks]), np.min(ppg_r[red_troughs])
        ir_high, ir_low = np.max(ppg_i[ir_peaks]), np.min(ppg_i[ir_troughs])

        ac_red = red_high - red_low
        ac_ir = ir_high - ir_low

        r = ( ac_red / red_low ) / ( ac_ir / ir_low )

        spo2 = 104 - (17 * r)
        if return_peaks:
            return spo2, r, dict(ir_peaks=ir_peaks, ir_troughs=ir_troughs, red_peaks=red_peaks, red_troughs=red_troughs)
        else:
            return spo2, r
    except Exception as e:
        print(f'Data rejected due to {e}')
        return 0, 0
