import numpy as np
from neurokit2.ppg.ppg_findpeaks import _ppg_findpeaks_bishop

def estimate_pulse_rate(ppg_red, ppg_ir, red_idx, ir_idx, fs):
    pulse_rate_multiplier = 60 / (len(ppg_red) / fs)
    pulse_rate = int(len(red_idx['peaks']) * pulse_rate_multiplier)
    return pulse_rate

def estimate_spo2(ppg_red: list, ppg_ir: list, red_idx, ir_idx):
    """Estimate absorbtion and SpO2.

    Args:
        ppg_red (list): PPG data (red LED).
        ppg_ir (list): PPG data (infrared LED).
        red_idx (dict): Peak data for ppg_red.
        ir_idx (dict): Peak data for ppg_ir.

    Returns:
        spo2 (float): SpO2 as a percentage.
        r (float): Absorption.
    """
    ppg_red = np.array(ppg_red)
    ppg_ir = np.array(ppg_ir)
    red_peaks, red_troughs = red_idx['peaks'], red_idx['troughs']
    red_high, red_low = np.max(ppg_red[red_peaks]), np.min(ppg_red[red_troughs])

    ir_peaks, ir_troughs = ir_idx['peaks'], ir_idx['troughs']
    ir_high, ir_low = np.max(ppg_ir[ir_peaks]), np.min(ppg_ir[ir_troughs])

    ac_red = red_high - red_low
    ac_ir = ir_high - ir_low

    r = ( ac_red / red_low ) / ( ac_ir / ir_low )
    spo2 = round(104 - (17 * r), 1)  # round to 1 decimal place
    return (spo2, r)
