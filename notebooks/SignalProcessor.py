import numpy as np
from wfdb import rdrecord
from Preprocessing.SignalLevelFiltering import bandpass, align_signals, get_similarity, get_snr, get_f0
from Preprocessing.BeatLevelFiltering import segment_beats, successive_beat_similarity, two_signal_beat_similarity
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
        self._similarity = []
        self._snr = []
        self._hr = []

    def run(self, low=0.5, high=8.0, sim1=0.6, sim2=0.9, snr_t=20, hr_diff=1/6, f0_low=0.667, f0_high=3.0):
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
        for i, f in enumerate(self._files):
            print('downloading')
            # Download data
            out = self._get_data(f)
            if out == False:
                continue
            else:
                pleth, abp = out[0], out[1]
            print('done')

            print('applying bandpass')
            # Apply bandpass filter to PLETH
            pleth = bandpass(pleth, low=low, high=high, fs=self._fs)
            print('done')

            overlap = int(self._fs / 2)
            l = self._win_len + overlap
            idx = window(pleth, l, overlap)

            print('getting valid windows')
            valid_windows = []
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
                if out != False:
                    valid_windows.append([out[0], out[1]])

            # TODO Beat level cleaning of windows
            # TODO Final dividing of windows before output
        return valid_windows, np.array(self._similarity), np.array(self._snr), np.array(self._hr)

    def _get_data(self, path):
        # Download
        response1 = download('https://' + path + '.hea')
        response2 = download('https://' + path + '.dat')
        if (response1 != 0) | (response2 != 0):
            return False
        else:
            # Extract signals from record
            rec = rdrecord(path)
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

        # Return valid PLETH and ABP window.
        return (p, a)

    def _beat_level_check(self):
        return
