import numpy as np
from wfdb import rdrecord
from Preprocessing.SignalLevelFiltering import bandpass, align_signals, get_similarity, get_hr, get_snr
from Preprocessing.BeatLevelFiltering import segment_beats, successive_beat_similarity, two_signal_beat_similarity
from Preprocessing.Utils import download, window, normalize


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
            pleth = bandpass(pleth)
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

    def _signal_level_check(self, p, a, sim1, sim2, snr_t, hr_diff):
        # Align signals in time
        p, a = align_signals(p, a, win_len=self._win_len)

        # Check time / spectral similarity
        time_sim = get_similarity(p, a)

        # Get magnitude of FFT for spectral similarity
        p_f = np.abs(np.fft.fft(p))
        a_f = np.abs(np.fft.fft(bandpass(a)))
        spec_sim = get_similarity(p_f, a_f)

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
