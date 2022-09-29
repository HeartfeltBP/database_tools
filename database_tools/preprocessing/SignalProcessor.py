import random
import numpy as np
import pandas as pd
from shutil import rmtree
from wfdb import rdrecord
from database_tools.preprocessing.SignalLevelFiltering import bandpass, align_signals, get_similarity, get_snr, get_f0
from database_tools.preprocessing.Utils import download, window, normalize


class SignalProcessor():
    def __init__(
        self,
        files,
        samples_per_patient,
        win_len,
        fs,
    ):
        self._files = files
        self._samples_per_patient = samples_per_patient
        self._win_len = win_len
        self._fs = fs

        # Metric tracking
        self._mrn     = []
        self._val     = []
        self._t_sim   = []
        self._f_sim   = []
        self._ppg_snr = []
        self._abp_snr = []
        self._ppg_hr  = []
        self._abp_hr  = []
        self._abp_min = []
        self._abp_max = []

    def run(self, config):
        """
        Process all signals in list of files. Performs signal and beat level cleaning.
        ppg output is bandpass filtered and normalized (standardized?).

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
        abp_min_bounds=config['abp_min_bounds']
        abp_max_bounds=config['abp_max_bounds']

        random.shuffle(self._files)
        mrn = 'start'
        for i, f in enumerate(self._files):
            last_mrn = mrn
            mrn = f.split('/')[-2]
            if last_mrn != mrn:
                n = 0  # int to count per patient samples
            elif n == self._samples_per_patient:
                continue

            # Download data
            out = self._get_data(f)
            if out == False:
                continue
            else:
                ppg, abp = out[0], out[1]

            # Apply bandpass filter to ppg
            ppg = bandpass(ppg, low=low, high=high, fs=self._fs)

            overlap = int(self._fs / 2)
            l = self._win_len + overlap
            idx = window(ppg, l, overlap)

            for i, j in idx:
                p = ppg[i:j]
                a = abp[i:j]

                # Signal level cleaning
                out = self._signal_level_check(
                    mrn=mrn,
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
                    abp_min_bounds=abp_min_bounds,
                    abp_max_bounds=abp_max_bounds,
                )

                # Add window if 'valid' is True
                if not out[0][1]:
                    self._append_metrics(out[0])
                else:
                    n += 1
                    self._append_metrics(out[0])
                    yield (out[1][0], out[1][1])

                # Move to next segment when patient fills up
                if n == self._samples_per_patient:
                    break
            rmtree('physionet.org/files/mimic3wdb/1.0/', ignore_errors=True)
        return

    def save_stats(self, path):
        df = pd.DataFrame(
            dict(
                mrn=self._mrn,
                valid=self._val,
                time_similarity=self._t_sim,
                spectral_similarity=self._f_sim,
                ppg_snr=self._ppg_snr,
                abp_snr=self._abp_snr,
                ppg_hr=self._ppg_hr,
                abp_hr=self._abp_hr,
                abp_min=self._abp_min,
                abp_max=self._abp_max,
            )
        )
        df.to_csv(path, index=False)
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
            ppg = rec.p_signal[:, signals.index('PLETH')].astype(np.float64)
            abp = rec.p_signal[:, signals.index('ABP')].astype(np.float64)

            # Set NaN to 0
            ppg[np.isnan(ppg)] = 0
            abp[np.isnan(abp)] = 0
        return (ppg, abp)

    def _signal_level_check(
        self,
        mrn,
        p,
        a,
        low,
        high,
        sim1,
        sim2,
        snr_t,
        hr_diff,
        f0_low,
        f0_high,
        abp_min_bounds,
        abp_max_bounds,
    ):
        # Align signals in time (output is win_len samples long)
        p, a = align_signals(p, a, win_len=self._win_len)

        # Get time similarity
        time_sim = get_similarity(p, a)

        # Get spectral similarity
        p_f = np.abs(np.fft.fft(p))
        a_f = np.abs(np.fft.fft(bandpass(a, low=low, high=high, fs=self._fs)))
        spec_sim = get_similarity(p_f, a_f)

        # Get SNR
        snr_p = get_snr(p, low=low, high=high, fs=self._fs)
        snr_a = get_snr(a, low=low, high=high, fs=self._fs)

        # Get fundamental frequencies
        f0_p = get_f0(p, fs=self._fs)
        f0_a = get_f0(a - np.mean(a), fs=self._fs)
        f0 = np.array([f0_p, f0_a])

        # Get min, max abp
        min_ = np.min(a)
        max_ = np.max(a)

        # Check similarity, snr, f0, and bp
        valid = True
        if (time_sim < sim1) | (spec_sim < sim1):
            valid = False
        elif (snr_p < snr_t) | (snr_a < snr_t):
            if (time_sim < sim2) | (spec_sim < sim2):
                valid = False
        elif np.abs(f0_p - f0_a) > hr_diff:
            if (time_sim < sim2) | (spec_sim < sim2):
                valid = False
        elif (f0 < f0_low).any() | (f0 > f0_high).any():
            valid = False
        elif ( ~np.isin(min_, abp_min_bounds).all() | ~np.isin(max_, abp_max_bounds) ):
            valid = False

        if valid:
            # Normalize ppg.
            p = normalize(p)
            return (
                [
                 mrn,
                 valid,
                 float(time_sim),
                 float(spec_sim),
                 float(snr_p),
                 float(snr_a),
                 float(f0_p * 60),
                 float(f0_a * 60),
                 float(min_),
                 float(max_)
                ],
                [p, a],
            )
        else:
            return (
                [
                 mrn,
                 valid,
                 float(time_sim),
                 float(spec_sim),
                 float(snr_p),
                 float(snr_a),
                 float(f0_p * 60),
                 float(f0_a * 60),
                 float(min_),
                 float(max_)
                ],
                [0, 0],
            )

    def _append_metrics(self, data):
        self._mrn.append(data[0])
        self._val.append(data[1])
        self._t_sim.append(data[2])
        self._f_sim.append(data[3])
        self._ppg_snr.append(data[4])
        self._abp_snr.append(data[5])
        self._ppg_hr.append(data[6])
        self._abp_hr.append(data[7])
        self._abp_min.append(data[8])
        self._abp_max.append(data[9])
