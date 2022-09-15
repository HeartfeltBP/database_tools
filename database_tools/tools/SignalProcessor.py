import numpy as np
from itertools import groupby
from sklearn.preprocessing import StandardScaler
from sklearn.utils import indexable
from scipy.signal import butter, cheby2, sosfiltfilt, medfilt
from heartpy.peakdetection import make_windows
from neurokit2.ppg import ppg_findpeaks
from heartpy.preprocessing import flip_signal
from database_tools.utils.compile import window_example


class SignalProcessor():
    def __init__(self, pleth, abp, mrn, fs=125, window_size=5):
        self._pleth = pleth
        self._abp = abp
        self._mrn = mrn
        self._fs = fs
        self._window_size = window_size

    def run(self):
        if indexable(self._pleth, self._abp) is None:
            raise ValueError('pleth and abp waveforms are not the same length')

        pleth_f = self._filter_pleth(self._pleth)
        self._abp[np.argwhere(np.isnan(self._abp))] = 0

        pleth_windows = self._window(pleth_f, self._window_size)
        abp_windows = self._window(self._abp, self._window_size)

        valid_samples = self._valid_windows(pleth_windows, abp_windows)
        n_samples = len(valid_samples)
        return valid_samples, n_samples

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
                            peaks,
                            valleys,
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
        n_peaks = len(peaks)
        if (n_peaks < (bpm_th1 / (60 / self._window_size))) | (n_peaks > (bpm_th2 / (60 / self._window_size))):
            return False

        # Waveform instability detection
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

    def _valid_abp_window(self,
                          sig,
                          peaks,
                          valleys,
                          flat_line_length=2):
        if (len(peaks) == 0) | (len(valleys) == 0):
            return False
        for i in peaks:
            try:
                seg = sig[i - flat_line_length:i + flat_line_length]
                if self._all_equal(seg):
                    return False
            except IndexError:
                continue
        return True

    def _calculate_bp(self, abp_win, peaks, valleys):
        sbp = np.mean(abp_win[peaks])
        dbp = np.mean(abp_win[valleys])
        return sbp, dbp

    def _valid_windows(self, pleth_windows, abp_windows):
        valid_samples = []
        for (pleth_win, abp_win) in zip(pleth_windows, abp_windows):
            try:
                pleth_peaks = ppg_findpeaks(pleth_win, sampling_rate=self._fs)['PPG_Peaks']
                pleth_valleys = ppg_findpeaks(flip_signal(pleth_win), sampling_rate=self._fs)['PPG_Peaks']
                abp_peaks = ppg_findpeaks(abp_win, sampling_rate=self._fs)['PPG_Peaks']
                abp_valleys = ppg_findpeaks(flip_signal(abp_win), sampling_rate=self._fs)['PPG_Peaks']
                if (self._valid_pleth_window(pleth_win,
                                             pleth_peaks,
                                             pleth_valleys,
                                             th1=-1.5,
                                             th2=1.5,
                                             dvc_th=1.0,
                                             bpm_th1=30,
                                             bpm_th2=220) & 
                    self._valid_abp_window(abp_win,
                                           abp_peaks,
                                           abp_valleys,
                                           flat_line_length=3)):
                    sbp, dbp = self._calculate_bp(abp_win, abp_peaks, abp_valleys)
                    example = window_example(pleth_win.astype(np.float64),
                                             sbp.astype(np.float64),
                                             dbp.astype(np.float64),
                                             self._mrn)
                    valid_samples.append(example)
            except:
                continue
        return valid_samples
