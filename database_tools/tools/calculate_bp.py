import numpy as np

def calculate_bp(abp_win, peaks, valleys):
    sbp = np.mean(abp_win[peaks])
    dbp = np.mean(abp_win[valleys])
    return sbp, dbp
