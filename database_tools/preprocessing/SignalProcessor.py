import vitaldb
import random
import numpy as np
import pandas as pd
from shutil import rmtree
from wfdb import rdrecord
from database_tools.preprocessing.SignalLevelFiltering import bandpass, align_signals, get_similarity, get_snr, flat_lines, beat_similarity
from database_tools.preprocessing.utils import download, window

import os
import pickle as pkl


class SignalProcessor():
    def __init__(
        self,
        partner,
        valid_df,
        win_len,
        fs,
        samples_per_patient,
    ):
        self._partner = partner
        self._valid_df = valid_df
        self._win_len = win_len
        self._fs = fs
        self._samples_per_patient = samples_per_patient

        # Metric tracking
        self._mrn     = []
        self._val     = []
        self._t_sim   = []
        self._f_sim   = []
        self._ppg_snr = []
        self._abp_snr = []
        self._ppg_hr  = []
        self._abp_hr  = []
        self._dbp = []
        self._sbp = []
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
        windowsize=config['windowsize']
        ma_perc=config['ma_perc']
        beat_sim=config['beat_sim']

        # Extract data from valid_df and shuffle in the same order
        patient_ids = self._valid_df['id']
        files = self._valid_df['url']
        temp = list(zip(patient_ids, files))
        random.shuffle(temp)
        patient_ids, files = zip(*temp)
        patient_ids, files = list(patient_ids), list(files)

        last_mrn = 'start'
        for i, (mrn, f) in enumerate(zip(patient_ids, files)):
            if last_mrn != mrn:
                n = 0  # int for counting samples per patient
            last_mrn = mrn

            # Download data
            out = self._get_data(f, self._partner)
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
                if n == self._samples_per_patient:
                    break
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
                    windowsize=windowsize,
                    ma_perc=ma_perc,
                    beat_sim=beat_sim,
                )

                # Add window if 'valid' is True
                if not out[0][1]:
                    self._append_metrics(out[0])
                else:
                    # if not os.path.exists('test-data.pkl'):
                    #     with open('test-data.pkl', 'wb') as f:
                    #         print('writing test data file')
                    #         pkl.dump(dict(ppg=ppg, abp=abp), f)
                    n += 1
                    self._append_metrics(out[0])
                    yield (out[1][0], out[1][1])

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
                sbp=self._sbp,
                dbp=self._dbp,
                ppg_beat_sim=self._ppg_beat_sim,
                abp_beat_sim=self._abp_beat_sim,
            )
        )
        df.to_csv(path, index=False)
        return

    def _get_data(self, path, partner):
        if partner == 'mimic3':
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
        elif partner == 'vital':
            data = vitaldb.load_case(path, ['ART', 'PLETH'], 1/125)
            abp, ppg = data[:, 0], data[:, 1]
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

        # Check for flat lines
        flat_p = flat_lines(p)
        flat_a = flat_lines(a)

        beat_sim_p = beat_similarity(
            p,
            windowsize=windowsize,
            ma_perc=ma_perc,
            fs=self._fs,
        )

        result = beat_similarity(
            a,
            windowsize=windowsize,
            ma_perc=ma_perc,
            fs=self._fs,
            get_bp=True,
        )
        if not isinstance(result, int):
            beat_sim_a, sbp, dbp = result
        else:
            beat_sim_a, sbp, dbp = result, 0, 0

        # Check similarity, snr, hr, bp, and peaks
        nan_check = np.nan in [time_sim, spec_sim, snr_p, snr_a, f0_p, f0_a, sbp, dbp]
        sim_check = (time_sim < sim) | (spec_sim < sim)
        snr_check = (snr_p < snr_t) | (snr_a < snr_t)
        hrdiff_check = np.abs(f0_p - f0_a) > hr_diff
        hr_check = (f0 < f0_low).any() | (f0 > f0_high).any()
        sbp_check = (sbp < abp_max_bounds[0]) | (sbp > abp_max_bounds[1])
        dbp_check = (dbp < abp_min_bounds[0]) | (dbp > abp_max_bounds[1])
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
                 float(dbp),
                 float(sbp),
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
                 float(dbp),
                 float(sbp),
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
        self._dbp.append(data[8])
        self._sbp.append(data[9])
        self._ppg_beat_sim.append(data[10])
        self._abp_beat_sim.append(data[11])
