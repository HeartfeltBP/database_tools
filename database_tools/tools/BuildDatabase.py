import json
import shutil
import random
import numpy as np
import pandas as pd
import vitaldb
from tqdm.notebook import tqdm
from wfdb import rdrecord
from database_tools.preprocessing.utils import ConfigMapper, download, window
from database_tools.preprocessing.functions import bandpass, align_signals
from database_tools.preprocessing.datastores import Window, MetricLogger, congruency_check


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

    def run(self, cm: ConfigMapper):
        logger = MetricLogger()

        print('Gettings valid segments...')
        patient_ids, files = self._get_valid_segs(self._data_dir + 'valid_segs.csv')
        partner = self._data_dir.split('/')[-2].split('-')[0]

        print('Collecting samples...')
        json_output = ''
        for mrn, f in zip(patient_ids, files):
            out = self._get_data(f, partner, cm)
            if out:
                ppg, abp = out[0], out[1]
            else:
                continue
            shutil.rmtree('physionet.org/files/mimic3wdb/1.0/', ignore_errors=True)

            # Apply bandpass filter to ppg
            ppg = bandpass(ppg, low=cm.freq_band[0], high=cm.freq_band[1], fs=cm.fs)

            overlap = int(cm.fs / 2)
            l = cm.win_len + overlap
            idx = window(ppg, l, overlap)

            print(f'Rejected samples: {logger.rejected_samples} --- Samples collected: {logger.valid_samples}')
            for i, j in tqdm(idx):
                p = ppg[i:j]
                a = abp[i:j]

                p, a = align_signals(p, a, win_len=cm.win_len, fs=cm.fs)

                # Evaluate ppg
                p_win = Window(p, cm, cm.checks)
                p_valid = p_win.valid

                # Evaluate abp
                a_win = Window(a, cm, cm.checks + ['bp'])
                a_valid = a_win.valid

                # Evaluate ppg vs abp
                time_sim, spec_sim, cong_check = congruency_check(p_win, a_win, cm)
                is_valid = p_valid & a_valid & cong_check

                logger.update_stats(mrn, is_valid, time_sim, spec_sim, p_win, a_win)  # update logger

                if is_valid:
                    json_output += json.dumps(dict(ppg=p.tolist(), abp=a.tolist())) + '\n'

                    # Write to file when count is reached. 
                    if (logger.valid_samples % self._samples_per_file) == 0:
                        file_name = str(int(logger.valid_samples / self._samples_per_file) - 1).zfill(3)
                        outfile = self._data_dir + f'data/lines/{partner}_{file_name}.jsonlines'
                        self._write_to_jsonlines(json_output, outfile)
                        json_output = ''
                        if logger.valid_samples >= self._max_samples:
                            logger.save_stats(self._data_dir + f'{partner}_stats.csv')
                            print('Done!')
                            return
                if logger.patient_samples == self._samples_per_patient:  # logger.mrn will update before this is reached again
                    break
        return

    def _get_valid_segs(self, valid_path):
        valid_df = pd.read_csv(valid_path)

        # Extract data from valid_df and shuffle in the same order
        temp = list(zip(valid_df['id'], valid_df['url']))
        random.shuffle(temp)
        patient_ids, files = [list(t) for t in zip(*temp)]
        patient_ids, files = list(patient_ids), list(files)
        return (patient_ids, files)

    def _get_data(self, path, partner, cm):
        if partner == 'mimic3':
            r1 = download(path + '.hea')
            r2 = download(path + '.dat')
            if (r1 != 0) | (r2 != 0):
                return False
            else:
                rec = rdrecord(path[8:])  # path w/o https://
                signals = rec.sig_name
                ppg = rec.p_signal[:, signals.index('PLETH')].astype(np.float64)
                abp = rec.p_signal[:, signals.index('ABP')].astype(np.float64)
                ppg[np.isnan(ppg)] = 0
                abp[np.isnan(abp)] = 0
            return (ppg, abp)
        elif partner == 'vital':
            data = vitaldb.load_case(path, ['ART', 'PLETH'], 1 / cm.fs)
            abp, ppg = data[:, 0], data[:, 1]
            ppg[np.isnan(ppg)] = 0
            abp[np.isnan(abp)] = 0
            return (ppg, abp)

    def _write_to_jsonlines(self, output, outfile):
        with open(outfile, 'w') as f:
            f.write(output)
