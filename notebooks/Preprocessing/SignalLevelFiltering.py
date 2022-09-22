import numpy as np

def align_signals(pleth, abp, win_len):
    fs = 125
    max_offset = int(fs / 2)

    corr = []
    for offset in range(0, max_offset):
        x = pleth[0 + offset : win_len + offset]
        corr.append(np.sum( x * abp ))
    idx = np.argmax(corr)
    x = pleth[0 + idx : win_len + idx]
    return (x, abp)

def get_similarity(x, y):
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
