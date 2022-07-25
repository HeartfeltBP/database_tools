import numpy as np
from itertools import groupby
from sklearn.preprocessing import StandardScaler
from sklearn.utils import indexable
from scipy.signal import butter, cheby2, sosfilt, sosfiltfilt, medfilt
from heartpy.peakdetection import make_windows
from neurokit2.ppg import ppg_findpeaks
from heartpy.preprocessing import flip_signal


class SignalProcessor():
    def __init__(self, pleth, abp, fs=125, window_size=5):
        self._pleth = pleth
        self._abp = abp
        self._fs = fs
        self._window_size = window_size

    def run(self):
        if indexable(self._pleth, self._abp) is None:
            raise ValueError('pleth and abp waveforms are not the same length')

        pleth_f = self._filter_pleth(self._pleth)

        pleth_windows = self._window(pleth_f, self._window_size)
        abp_windows = self._window(self._abp, self._window_size)

        valid_windows = self._valid_windows(pleth_windows, abp_windows)
        # ...
        return abp_windows, valid_windows

    def _filter_pleth(self, sig):
        scaler = StandardScaler()
        btr = butter(4, [0.5, 8.0], btype='bandpass', output='sos', fs=self._fs)
        cby = cheby2(4, 20, [0.5, 8.0], btype='bandpass', output='sos', fs=self._fs)

        sig[np.argwhere(np.isnan(sig))] = 0  # Set nan to 0
        sig = scaler.fit_transform(sig.reshape(-1, 1)).reshape(-1)
        sig = sosfiltfilt(btr, sig, padtype=None)
        sig = sosfiltfilt(cby, sig, padtype=None)
        sig = medfilt(sig, kernel_size=3)
        return sig

    def _window(self, sig, window_size):
        idx = make_windows(sig, sample_rate=self._fs, windowsize=window_size, min_size=(window_size * self._fs))
        windows = np.array([np.array(sig[i:j]) for i, j in idx])
        return windows

    def _valid_pleth_window(self,
                            sig,
                            th1=-1.5,
                            th2=1.5,
                            dvc_th=1.0,
                            bpm_th1=30,
                            bpm_th2=220):
        """
        th1 : float
            Lower amplitude threshold.

        th2 : float
            Ppper amplitude threshold.

        dvc_th : float
            Dynamic variation coefficient threshold.

        bpm_th1 : int
            Lower bpm threshold.

        bpm_th2 : int
            Upper bpm threshold.
        """
        # TODO:

        # Pulse interference detection
        if (sig < th1).any() | (sig > th2).any():
            return False

        # Missing segment detection
        mj = np.mean(abs(sig))
        vj = np.mean(np.square(sig - mj))
        dvc = vj / mj
        if dvc < dvc_th:
            return False

        # Motion artifact detection
        peaks = ppg_findpeaks(sig, sampling_rate=self._fs)['PPG_Peaks']
        n_peaks = len(peaks)
        if (n_peaks < (bpm_th1 / (60 / self._window_size))) | (n_peaks > (bpm_th2 / (60 / self._window_size))):
            return False

        # Waveform instability detection
        valleys = ppg_findpeaks(flip_signal(sig), sampling_rate=self._fs)['PPG_Peaks']

        if peaks[0] > valleys[0]:  # Is the first index a peak or valley?
            x = valleys
            y = peaks
        else:
            x = peaks
            y = valleys

        for i, idx in enumerate(x):  # Are peaks and valleys in index order?
            try:
                jdx = x[i + 1]
            except IndexError:
                continue
            if (y[i] < idx) | (y[i] > jdx):
                return False
        return True

    def _all_equal(self, iterable):
        g = groupby(iterable)
        return next(g, True) and not next(g, False)

    def _valid_abp_window(self, sig, flat_line_length=2):
        peaks = ppg_findpeaks(sig, sampling_rate=self._fs)['PPG_Peaks']
        for i in peaks:
            try:
                seg = sig[i - flat_line_length:i + flat_line_length]
                if self._all_equal(seg):
                    return False
            except IndexError:
                continue
        return True

    def _valid_windows(self, pleth_windows, abp_windows):
        valid_idx = []
        for i, (pleth_win, abp_win) in enumerate(zip(pleth_windows, abp_windows)):
            if (self._valid_pleth_window(pleth_win,
                                         th1=-1.5,
                                         th2=1.5,
                                         dvc_th=1.0,
                                         bpm_th1=30,
                                         bpm_th2=220) & 
                self._valid_abp_window(abp_win)):

                valid_idx.append(i)
        
        valid_pleth_windows = pleth_windows[valid_idx, :]
        valid_abp_windows = abp_windows[valid_idx, :]
        return (valid_pleth_windows, valid_abp_windows)
