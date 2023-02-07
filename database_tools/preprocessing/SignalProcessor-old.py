import vitaldb
import random
import typing
import numpy as np
import pandas as pd
from shutil import rmtree
from wfdb import rdrecord
from database_tools.preprocessing.SignalLevelFiltering import bandpass, align_signals, get_similarity, get_snr, flat_lines, beat_similarity
from database_tools.preprocessing.utils import download, window

import os
import pickle as pkl

from dataclasses import dataclass, InitVar


@dataclass(frozen=True)
class ConfigMapper:

    config: InitVar[dict]

    def __post_init__(self, config: dict):
        for key, value in config.items():
            object.__setattr__(self, key, value)


class SignalProcessor():
    def __init__(
        self,
        partner: str,
        valid_df: pd.DataFrame,
        win_len: int,
        fs: int,
        samples_per_patient: int,
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

    def run(self, cm: ConfigMapper) -> typing.Tuple[np.ndarray, np.ndarray]:
        """
        Generator that processes all signals in list of files. Performs signal 
        and level filtering and cleaning and yields valid samples (ppg, abp window).

        Args:
            cm (ConfigMapper): Config mapping dataclass for the following:

                freq_band (List[float]): Bandpass frequencies.
                sim (float): Time & spectral similarity threshold (minimum).
                snr (float): SNR threshold (minimum).
                hr_freq_band (List[float]): Valid heartrate frequency band in Hz.
                hr_delta (int): Maximum HR difference between PPG and ABP signals.
                abp_min_bounds (List[int]): Upper and lower bounds for diastolic bp.
                abp_max_bounds (List[int]): Upper and lower bounds for diastolic bp.
                windowsize (int): Windowsize for rolling mean.
                ma_perc (int): Multiplier for peak detection.
                beat_sim (float): Lower threshold for beat similarity.

        Returns:
            ppg (np.ndarray): Valid ppg window.
            abp (np.ndarray): Valid abp window.
        """
        # Extract data from valid_df and shuffle in the same order
        temp = list(zip(self._valid_df['id'], self._valid_df['url']))
        random.shuffle(temp)
        patient_ids, files = [list(t) for t in zip(*temp)]
        patient_ids, files = list(patient_ids), list(files)

        last_mrn = 'start'
        for _, (mrn, f) in enumerate(zip(patient_ids, files)):
            if last_mrn != mrn:
                n = 0  # int for counting samples per patient
            last_mrn = mrn

            # Download data
            out = self._get_data(f, self._partner)
            if out:
                ppg, abp = out[0], out[1]

            # Apply bandpass filter to ppg
            ppg = bandpass(ppg, low=cm.freq_band[0], high=cm.freq_band[1], fs=self._fs)

            overlap = int(self._fs / 2)
            l = self._win_len + overlap
            idx = window(ppg, l, overlap)

            for i, j in idx:
                if n == self._samples_per_patient:
                    break
                p = ppg[i:j]
                a = abp[i:j]

                # Signal level cleaning
                out = self._signal_level_check(mrn, p, a, cm)

                # Add window if 'valid' is True
                if not out[0][1]:
                    self._append_metrics(out[0])
                else:
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

    def _signal_level_check(self, mrn, p, a, cm):
        # Align signals in time (output is win_len samples long)
        p, a = align_signals(p, a, win_len=self._win_len, fs=self._fs)

        # Get time similarity
        time_sim = get_similarity(p, a)

        # Get spectral similarity
        p_f = np.abs(np.fft.fft(p))
        a_f = np.abs(np.fft.fft(bandpass(a, low=cm.freq_band[0], high=cm.freq_band[1], fs=self._fs)))
        spec_sim = get_similarity(p_f, a_f)

        # Get SNR & fundamental frequencies
        snr_p, f0_p = get_snr(p, low=cm.freq_band[0], high=cm.freq_band[1], df=0.2, fs=self._fs)
        snr_a, f0_a = get_snr(a, low=0, high=self._fs / 2, df=0.2, fs=self._fs)
        f0 = np.array([f0_p, f0_a])

        # Check for flat lines
        flat_p = flat_lines(p)
        flat_a = flat_lines(a)

        beat_sim_p = beat_similarity(
            p,
            windowsize=cm.windowsize,
            ma_perc=cm.ma_perc,
            fs=self._fs,
        )

        result = beat_similarity(
            a,
            windowsize=cm.windowsize,
            ma_perc=cm.ma_perc,
            fs=self._fs,
            get_bp=True,
        )
        if not isinstance(result, int):
            beat_sim_a, sbp, dbp = result
        else:
            beat_sim_a, sbp, dbp = result, 0, 0

        # Check similarity, snr, hr, bp, and peaks
        nan_check = np.nan in [time_sim, spec_sim, snr_p, snr_a, f0_p, f0_a, sbp, dbp]
        sim_check = (time_sim < cm.sim) | (spec_sim < cm.sim)
        snr_check = (snr_p < cm.snr) | (snr_a < cm.snr)
        hrdiff_check = np.abs(f0_p - f0_a) > cm.hr_delta
        hr_check = (f0 < cm.hr_freq_band[0]).any() | (f0 > cm.hr_freq_band[1]).any()
        sbp_check = (sbp < cm.abp_max_bounds[0]) | (sbp > cm.abp_max_bounds[1])
        dbp_check = (dbp < cm.abp_min_bounds[0]) | (dbp > cm.abp_max_bounds[1])
        beat_check = (beat_sim_p < cm.beat_sim) | (beat_sim_a < cm.beat_sim)
        if nan_check | sim_check | snr_check | hrdiff_check | hr_check | dbp_check | sbp_check | flat_p | flat_a | beat_check:
            valid = False
        else:
            valid = True

        stats = [
                 mrn,
                 valid,
                 float(time_sim),
                 float(spec_sim),
                 float(snr_p),
                 float(snr_a),
                 float(f0_p * 60),  # multiply by 60 to estimate HR
                 float(f0_a * 60),
                 float(dbp),
                 float(sbp),
                 float(beat_sim_p),
                 float(beat_sim_a),
        ]
        if valid:
            return (stats, [p, a])
        else:
            return (stats, [0, 0])

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
