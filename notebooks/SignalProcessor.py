import numpy as np
from wfdb import rdrecord
from Preprocessing.SignalLevelFiltering import align_signals, get_similarity, get_hr, get_snr
from Preprocessing.BeatLevelFiltering import segment_beats, successive_beat_similarity, two_signal_beat_similarity
from Preprocessing.Utils import download, bandpass, window, normalize


class SignalProcessor():
    def __init__(
        self,
        files,
        win_len,
        fs=125,
    ):
        self._files = files
        self._win_len = win_len
        self._fs = fs

    def run(self, sim1=0.6, sim2=0.9, snr_t=2, hr_diff=10):
        for i, f in enumerate(self._files):
            # Download data
            out = self._get_data(f)

            if out == False:
                continue
            else:
                pleth, abp = out[0], out[1]

            # Apply bandpass filter to PLETH
            pleth = bandpass(pleth)

            overlap = self._fs / 2
            pleth_win = window(pleth, self._win_len, overlap)
            abp_win = window(abp, self._win_len, overlap)
            print(pleth_win.shape)
            print(abp_win.shape)
            
            valid_windows = []
            for p, a in zip(pleth_win, abp_win):
                # Signal level cleaning
                out = self._signal_level_check(p, a, sim1, sim2, snr_t, hr_diff)

                # Add window if valid
                if out != False:
                    valid_windows.append([out[0], out[1]])
            
            # TODO Beat level cleaning of windows
            # TODO Final dividing of windows before output
        return valid_windows

    def _get_data(self, path):
        # Download
        response1 = download('https://' + path + '.hea')
        response2 = download('https://' + path + '.dat')
        if (response1 != 0) | (response2 != 0):
            return False

        # Extract signals from record
        rec = rdrecord(path)
        signals = rec.sig_name
        pleth = rec.p_signal[:, signals.index('PLETH')].astype(np.float64)
        abp = rec.p_signal[:, signals.index('ABP')].astype(np.float64)

        # Set NaN to 0
        pleth[np.isnan(pleth)] = 0
        abp[np.isnan(abp)] = 0
        return (pleth, abp)

    def _signal_level_check(self, p, a, sim1, sim2, snr_t, hr_diff):
        # Align signals in time
        p, a = align_signals(p, a)

        # Check time / spectral similarity
        time_sim = get_similarity(p, a, spectral=False)
        spec_sim = get_similarity(p, a, fft=True)
        if (time_sim < sim1) | (spec_sim < sim1):
            return False

        # TODO Implement SNR function
        # Check SNR
        # snr = get_snr(p, a)
        # if (snr < snr_t).any():
        #     if (time_sim < sim2) | (spec_sim < sim2):
        #         return False

        # TODO Implement HR function
        # Check HR difference
        # hr = get_hr(p, a)
        # if np.abs(hr[0] - hr[1]) > hr_diff:
        #     if (time_sim < sim2) | (spec_sim < sim2):
        #         return False
        return (p, a)
