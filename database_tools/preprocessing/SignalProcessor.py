import random
import numpy as np
import pandas as pd
from shutil import rmtree
from wfdb import rdrecord
from neurokit2.ppg import ppg_findpeaks
from heartpy.preprocessing import flip_signal
from database_tools.preprocessing.SignalLevelFiltering import bandpass, align_signals, get_similarity, get_snr, flat_lines, beat_similarity
from database_tools.preprocessing.Utils import download, window


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
        self._ppg_beat_sim = []
        self._abp_beat_sim = []

    def run(self, config):
        """
        Generator that processes all signals in list of files. Performs signal 
        and level filtering and cleaning and yields valid samples (ppg, abp window).

        Args:
            low (float): Low frequency for bandpass filtering.
            high (float): High frequency for bandpass filtering.
            sim (float): Time & spectral similarity threshold (minimum).
            snr_t (float): SNR threshold (minimum).
            hr_diff (int): Maximum HR difference between PPG and ABP signals.
            f0_low (float): Minimum valid HR in Hz.
            f0_high (float): Maximum valid HR in Hz.
            abp_min_bounds (List[int]): Upper and lower bounds for DBP.
            abp_max_bounds (List[int]): Upper and lower bounds for SBP.

        Returns:
            ppg, abp (np.ndarray): Valid PPG, ABP window pair.
        """
        low=config['low']
        high=config['high']
        sim=config['sim']
        df=config['df']
        snr_t=config['snr_t']
        hr_diff=config['hr_diff']
        f0_low=config['f0_low']
        f0_high=config['f0_high']
        abp_min_bounds=config['abp_min_bounds']
        abp_max_bounds=config['abp_max_bounds']
        pp_min=config['pp_min']
        pp_max=config['pp_max']
        n_peaks=config['n_peaks']
        windowsize=config['windowsize']
        ma_perc=config['ma_perc']
        beat_sim=config['beat_sim']

        random.shuffle(self._files)
        mrn = 'start'
        num_windows_completed = 0
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
                    sim=sim,
                    df=df,
                    snr_t=snr_t,
                    hr_diff=hr_diff,
                    f0_low=f0_low,
                    f0_high=f0_high,
                    abp_min_bounds=abp_min_bounds,
                    abp_max_bounds=abp_max_bounds,
                    pp_min=pp_min,
                    pp_max=pp_max,
                    n_peaks=n_peaks,
                    windowsize=windowsize,
                    ma_perc=ma_perc,
                    beat_sim=beat_sim,
                )
                # num_windows_completed += 1
                # if num_windows_completed % 1000 == 0:
                #     print(f'Done processing {num_windows_completed} windows')
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
                ppg_beat_sim=self._ppg_beat_sim,
                abp_beat_sim=self._abp_beat_sim,
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
        sim,
        df,
        snr_t,
        hr_diff,
        f0_low,
        f0_high,
        abp_min_bounds,
        abp_max_bounds,
        pp_min,
        pp_max,
        n_peaks,
        windowsize,
        ma_perc,
        beat_sim,
    ):
        # Align signals in time (output is win_len samples long)
        p, a = align_signals(p, a, win_len=self._win_len, fs=self._fs)

        # Get time similarity
        time_sim = get_similarity(p, a)

        # Get spectral similarity
        p_f = np.abs(np.fft.fft(p))
        a_f = np.abs(np.fft.fft(bandpass(a, low=low, high=high, fs=self._fs)))
        spec_sim = get_similarity(p_f, a_f)

        # Get SNR & fundamental frequencies
        snr_p, f0_p = get_snr(p, low=low, high=high, df=df, fs=self._fs)
        snr_a, f0_a = get_snr(a, low=0, high=self._fs / 2, df=df, fs=self._fs)
        f0 = np.array([f0_p, f0_a])

        # Get min, max abp
        min_ = np.min(a)
        max_ = np.max(a)

        # Check for flat lines
        flat_p = flat_lines(p)
        flat_a = flat_lines(a)

        beat_sim_p = beat_similarity(
            p,
            n_peaks=n_peaks,
            windowsize=windowsize,
            ma_perc=ma_perc,
            fs=self._fs
        )

        beat_sim_a = beat_similarity(
            a,
            n_peaks=n_peaks,
            windowsize=windowsize,
            ma_perc=ma_perc,
            fs=self._fs
        )

        # Check similarity, snr, hr, bp, and peaks
        nan_check = np.nan in [time_sim, spec_sim, snr_p, snr_a, f0_p, f0_a, min_, max_]
        sim_check = (time_sim < sim) | (spec_sim < sim)
        snr_check = (snr_p < snr_t) | (snr_a < snr_t)
        hrdiff_check = np.abs(f0_p - f0_a) > hr_diff
        hr_check = (f0 < f0_low).any() | (f0 > f0_high).any()
        dbp_check = (min_ < abp_min_bounds[0]) | (max_ > abp_max_bounds[1])
        sbp_check = (max_ < abp_max_bounds[0]) | (max_ > abp_max_bounds[1])
        # TODO Is this necessary?
        # pp_check = (max_ - min_ < pp_min) | (max_ - min_ > pp_max)
        beat_check = (beat_sim_p < beat_sim) | (beat_sim_a < beat_sim)
        if nan_check | sim_check | snr_check | hrdiff_check | hr_check | dbp_check | sbp_check | flat_p | flat_a | beat_check:
            valid = False
        else:
            valid = True

        if valid:
            return (
                [
                 mrn,
                 valid,
                 float(time_sim),
                 float(spec_sim),
                 float(snr_p),
                 float(snr_a),
                 float(f0_p * 60),  # Multiply by 60 to estimate HR
                 float(f0_a * 60),
                 float(min_),
                 float(max_),
                 float(beat_sim_p),
                 float(beat_sim_a),
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
                 float(max_),
                 float(beat_sim_p),
                 float(beat_sim_a),
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
        self._ppg_beat_sim.append(data[10])
        self._abp_beat_sim.append(data[11])
