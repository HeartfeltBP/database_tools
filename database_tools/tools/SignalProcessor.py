import os
import numpy as np
from itertools import groupby
from sklearn.preprocessing import StandardScaler
from sklearn.utils import indexable
from wfdb import rdrecord
from scipy.signal import butter, cheby2, sosfiltfilt, medfilt
from heartpy.peakdetection import make_windows
from neurokit2.ppg import ppg_findpeaks
from heartpy.preprocessing import flip_signal


class SignalProcessor():
    def __init__(self, segments, folder, mrn, data_dir, fs=125, window_size=5):
        self._segments = segments
        self._folder = folder
        self._mrn = mrn
        self._data_dir = data_dir
        self._fs = fs
        self._window_size = window_size
        self._sample_index = 0

    def sample_generator(self):
        for seg in self._segments:
            pleth, abp = self._get_sigs(self._folder, seg)
            if indexable(pleth, abp) is None:
                raise ValueError('pleth and abp waveforms are not the same length')

            pleth_f = self._filter_pleth(pleth)
            abp[np.argwhere(np.isnan(abp))] = 0

            pleth_windows = self._window(pleth_f, self._window_size)
            abp_windows = self._window(abp, self._window_size)

            valid_samples = self._valid_windows(pleth_windows, abp_windows)
            for sample in valid_samples:
                yield sample

    def _download(self, path):
        response = os.system(f'wget -q -r -np {path}')
        return response

    def _get_sigs(self, folder, seg):
        path = self._data_dir + folder + seg
        response = self._download(path + '.dat')
        if response == 0:
            rec = rdrecord(path)
            signals = rec.sig_name
            pleth = rec.p_signal[:, signals.index('PLETH')].astype(np.float64)
            abp = rec.p_signal[:, signals.index('ABP')].astype(np.float64)
            return pleth, abp
        return None, None

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
                    sample = dict(
                        pleth=list(pleth_win),
                        abp=list(abp_win),
                        sample_id=str(str(self._mrn) + f'_{str(self._sample_index).zfill(8)}'),
                    )
                    valid_samples.append(sample)
                    self._sample_index += 1
                    if self._sample_index == 2000:
                        return valid_samples
            except:
                continue
        return valid_samples
