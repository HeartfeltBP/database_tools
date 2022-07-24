import numpy as np
from sklearn.utils import indexable
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, cheby2, sosfiltfilt, medfilt
from heartpy.peakdetection import make_windows
from neurokit2.ppg import ppg_findpeaks
from heartpy.preprocessing import flip_signal


class SignalProcessor():
    def __init__(self, pleth, abp, fs=125):
        self._pleth = pleth
        self._abp = abp
        self._fs = fs

    def run(self):
        if indexable(self._pleth, self._abp) is None:
            raise ValueError('pleth and abp waveforms are not the same length')

        pleth_windows = self._process(self._pleth)
        abp_windows = self._process(self._abp)
        valid_windows = self._valid_windows(pleth_windows, abp_windows)
        # ...
        return

    def _window(self, sig, window_size=5):
        idx = make_windows(sig, sample_rate=self._fs, windowsize=window_size, min_size=(window_size * self.fs))
        windows = np.array([np.array(sig[i:j]) for i, j in idx])
        return windows

    def _process(self, sig):
        scaler = StandardScaler()
        btr = butter(4, [0.5, 8.0], btype='bandpass', output='sos', fs=self._fs)
        cby = cheby2(4, 20, [0.5, 8.0], btype='bandpass', output='sos', fs=self._fs)

        sig[np.argwhere(np.isnan(sig))] = 0  # Set nan to 0
        sig = scaler.fit_transform(sig.reshape(-1, 1)).reshape(-1)
        sig = sosfiltfilt(btr, sig, padtype=None)
        sig = sosfiltfilt(cby, sig, padtype=None)
        sig = medfilt(sig, kernel_size=3)
        windows = self._window(sig)
        return windows

    def _valid_pleth_window(self, sig):
        # TODO:
        return

    def _valid_abp_window(self, sig):
        # TODO:
        return

    def _valid_windows(self, pleth_windows, abp_windows):
        X = pleth_windows
        Y = abp_windows

        valid_idx = []
        for i, X_win, Y_win in enumerate(zip(X, Y)):
            if self._valid_pleth_window(X_win) & self._valid_abp_window(Y_win):
                valid_idx.append(i)
        return
