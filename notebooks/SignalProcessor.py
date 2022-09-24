from shutil import rmtree
import numpy as np
from wfdb import rdrecord
from Preprocessing.SignalLevelFiltering import bandpass, align_signals, get_similarity, get_snr, get_f0
from Preprocessing.Utils import download, window, normalize


class SignalProcessor():
    def __init__(
        self,
        files,
        win_len=1024,
        fs=125,
    ):
        self._files = files
        self._win_len = win_len
        self._fs = fs
        
        # For testing
        self._n_excluded = 0
        self._similarity = []
        self._snr = []
        self._hr = []
        self._abp_min = 999
        self._abp_max = 0

    def run(self, config):
        """
        Process all signals in list of files. Performs signal and beat level cleaning.
        PLETH output is bandpass filtered and normalized (standardized?).

        Args:
            low (float, optional): _description_. Defaults to 0.5.
            high (float, optional): _description_. Defaults to 8.0.
            sim1 (float, optional): _description_. Defaults to 0.6.
            sim2 (float, optional): _description_. Defaults to 0.9.
            snr_t (int, optional): _description_. Defaults to 20.
            hr_diff (_type_, optional): _description_. Defaults to 1/6.
            f0_low (float, optional): _description_. Defaults to 0.667.
            f0_high (float, optional): _description_. Defaults to 3.0.

        Returns:
            None
        """
        low=config['low']
        high=config['high']
        sim1=config['sim1']
        sim2=config['sim2']
        snr_t=config['snr_t']
        hr_diff=config['hr_diff']
        f0_low=config['f0_low']
        f0_high=config['f0_high']

        for i, f in enumerate(self._files):
            # Download data
            out = self._get_data(f)
            if out == False:
                continue
            else:
                pleth, abp = out[0], out[1]

            # Apply bandpass filter to PLETH
            pleth = bandpass(pleth, low=low, high=high, fs=self._fs)

            overlap = int(self._fs / 2)
            l = self._win_len + overlap
            idx = window(pleth, l, overlap)

            for i, j in idx:
                p = pleth[i:j]
                a = abp[i:j]
                
                # Signal level cleaning
                out = self._signal_level_check(
                    p=p,
                    a=a,
                    low=low,
                    high=high,
                    sim1=sim1,
                    sim2=sim2,
                    snr_t=snr_t,
                    hr_diff=hr_diff,
                    f0_low=f0_low,
                    f0_high=f0_high,
                )

                # Add window if valid
                if not out:
                    self._n_excluded += 1
                else:
                    yield (out[0], out[1])
            rmtree('physionet.org/files/mimic3wdb/1.0/', ignore_errors=True)

            # TODO Beat level cleaning of windows
            # TODO Final dividing of windows before output
        return

    def _get_data(self, path):
        # Download
        response1 = download(path + '.hea')
        response2 = download(path + '.dat')
        if (response1 != 0) | (response2 != 0):
            return False
        else:
            # Extract signals from record
            rec = rdrecord(path[8:])  # cut of https:// from path
            signals = rec.sig_name
            pleth = rec.p_signal[:, signals.index('PLETH')].astype(np.float64)
            abp = rec.p_signal[:, signals.index('ABP')].astype(np.float64)

            # Set NaN to 0
            pleth[np.isnan(pleth)] = 0
            abp[np.isnan(abp)] = 0
        return (pleth, abp)

    def _signal_level_check(
        self,
        p,
        a,
        low,
        high,
        sim1,
        sim2,
        snr_t,
        hr_diff,
        f0_low,
        f0_high
    ):
        # Align signals in time (output is win_len samples long)
        p, a = align_signals(p, a, win_len=self._win_len)

        # Check time / spectral similarity
        time_sim = get_similarity(p, a)

        # Get magnitude of FFT for spectral similarity
        p_f = np.abs(np.fft.fft(p))
        a_f = np.abs(np.fft.fft(bandpass(a, low=low, high=high, fs=self._fs)))
        spec_sim = get_similarity(p_f, a_f)

        self._similarity.append(np.array([time_sim, spec_sim]))

        if (time_sim < sim1) | (spec_sim < sim1):
            return False

        # Check SNR
        snr = np.array(
            [
                get_snr(p, low=low, high=high, fs=self._fs),
                get_snr(a, low=low, high=high, fs=self._fs),
            ]
        )

        self._snr.append(snr)
        if (snr < snr_t).any():
            if (time_sim < sim2) | (spec_sim < sim2):
                return False

        # Check HR thresholds & difference
        hr = np.array(
            [
                get_f0(p, fs=self._fs),
                get_f0(a - np.mean(a), fs=self._fs),
            ]
        )
        self._hr.append(hr)
        if np.abs(hr[0] - hr[1]) > hr_diff:
            if (time_sim < sim2) | (spec_sim < sim2):
                return False
        if (hr < f0_low).any() | (hr > f0_high).any():
            return False

        # Normalize pleth.
        p = normalize(p)

        # Update min, max abp
        _min = np.min(a)
        _max = np.max(a)
        if _min < self._abp_min:
            self._abp_min = _min
        if _max > self._abp_max:
            self._abp_max = _max
        return (p, a)

    def get_stats(self):
        return self._n_excluded, np.array(self._similarity), np.array(self._snr), np.array(self._hr), self._abp_max, self._abp_min
