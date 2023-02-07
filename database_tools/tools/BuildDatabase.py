import sys
import json
import shutil
import random
import numpy as np
import pandas as pd

import vitaldb
from wfdb import rdrecord
from database_tools.preprocessing.utils import download, window
from database_tools.preprocessing.SignalLevelFiltering import bandpass, align_signals

from database_tools.preprocessing.SignalProcessor import ConfigMapper, Window, congruency_check


class BuildDatabase():
    def __init__(
        self,
        data_dir,
        samples_per_file=2500,
        samples_per_patient=500,
        max_samples=300000,
    ) -> None:
        self._data_dir = data_dir
        self._samples_per_file = samples_per_file
        self._samples_per_patient = samples_per_patient
        self._max_samples = max_samples

        # Metric tracking
        self.mrn     = []
        self.val     = []
        self.t_sim   = []
        self.f_sim   = []
        self.ppg_snr = []
        self.abp_snr = []
        self.ppg_hr  = []
        self.abp_hr  = []
        self.dbp = []
        self.sbp = []
        self.ppg_beat_sim = []
        self.abp_beat_sim = []

    def run(self, cm: ConfigMapper):
        patient_ids, files = self._get_valid_segs(self._data_dir + 'valid_segs.csv')
        partner = self._data_dir.split('-')[0]

        total_samples = 0
        json_output = ''
        prev_mrn = ''
        for mrn, f in zip(patient_ids, files):
            if prev_mrn != mrn:
                patient_samples = 0  # samples per patient
                prev_mrn = mrn

            out = self._get_data(f, partner)
            if out:
                ppg, abp = out[0], out[1]

            # Apply bandpass filter to ppg
            ppg = bandpass(ppg, low=cm.freq_band[0], high=cm.freq_band[1], fs=cm.fs)

            overlap = int(cm.fs / 2)
            l = cm.win_len + overlap
            idx = window(ppg, l, overlap)

            for i, j in idx:
                if patient_samples == self._samples_per_patient:
                    break
                p = ppg[i:j]
                a = abp[i:j]

                p, a = align_signals(p, a, win_len=cm.win_len, fs=cm.fs)

                p_win = Window(p, cm, ['snr', 'hr', 'flat', 'beat'])
                p_valid = p_win.valid
                a_win = Window(a, cm, ['snr', 'hr', 'flat', 'beat', 'bp'])
                a_valid = a_win.valid
                self._append_metrics(p_win, a_win)

                # congruency check performs similarity and hr delta checks
                if p_valid & a_valid & congruency_check(p_win, a_win, cm):
                    samples += json.dumps(dict(ppg=ppg.tolist(), abp=abp.tolist())) + '\n'
                    total_samples += 1
                    sys.stdout.flush()
                    sys.stdout.write('%s\r' % str(total_samples))

                    # Write to file when count is reached. 
                    if (total_samples % self._samples_per_file) == 0:
                        fn = str(int(total_samples / self._samples_per_file) - 1).zfill(3)  # file name
                        outfile = self._data_dir + f'data/lines/{partner}_{fn}.jsonlines'
                        self._write_to_jsonlines(json_output, outfile)
                        json_output = ''
                        if total_samples >= self._max_samples:
                            return

            # Delete data when done with a patient
            shutil.rmtree('physionet.org/files/mimic3wdb/1.0/', ignore_errors=True)
        return

    def _get_valid_segs(self, valid_path):
        valid_df = pd.read_csv(valid_path)

        # Extract data from valid_df and shuffle in the same order
        temp = list(zip(valid_df['id'], valid_df['url']))
        random.shuffle(temp)
        patient_ids, files = [list(t) for t in zip(*temp)]
        patient_ids, files = list(patient_ids), list(files)
        return (patient_ids, files)

    def _get_data(self, path, partner):
        if partner == 'mimic3':
            response1 = download(path + '.hea')
            response2 = download(path + '.dat')
            if (response1 != 0) | (response2 != 0):
                return False
            else:
                rec = rdrecord(path[8:])  # path w/o https://
                signals = rec.sig_name
                ppg = rec.p_signal[:, signals.index('PLETH')].astype(np.float64)
                abp = rec.p_signal[:, signals.index('ABP')].astype(np.float64)
        elif partner == 'vital':
            data = vitaldb.load_case(path, ['ART', 'PLETH'], 1/125)
            abp, ppg = data[:, 0], data[:, 1]
        ppg[np.isnan(ppg)] = 0
        abp[np.isnan(abp)] = 0
        return (ppg, abp)

    def _append_metrics(self, ppg: Window, abp: Window):
        return

    def _write_to_jsonlines(self, output, outfile):
        with open(outfile, 'w') as f:
            f.write(output)

    def save_stats(self):
        return
