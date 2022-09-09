import os
import numpy as np
import pandas as pd
from wfdb import rdheader, rdrecord
from shutil import rmtree
from tqdm.notebook import tqdm
from .SignalProcessor import SignalProcessor
from .compile import append_patient, append_sample_count


class BuildDatabase():
    def __init__(self,
                 records_path,
                 data_profile_csv,
                 pleth_csv,
                 abp_csv,
                 max_records,
                 data_dir='physionet.org/files/mimic3wdb/1.0/'):
        self._records_path = records_path
        self._data_profile_csv = data_profile_csv
        self._pleth_csv = pleth_csv
        self._abp_csv = abp_csv
        self._max_records = max_records
        self._data_dir = data_dir

    def run(self):
        # Load records.
        records = pd.read_csv(self._records_path, names=['patient_dir'])

        # Get last record (MRN) added to data profile.
        last_record = str(self._get_last_record(self._data_profile_csv))

        start = 0  # Start idx is 0 unless database is partially built.
        if last_record != 'test':
            start = records.index[records['patient_dir'].str.contains(last_record)].tolist()[0] + 1

        num_processed = 0
        for folder in tqdm(records['patient_dir'][start::]):
            if num_processed == self._max_records:
                break
            self._extract(folder)
            num_processed += 1
        return

    def _get_last_record(self, path):
        df = pd.read_csv(path, index_col='index')
        last_record = df['mrn'].iloc[-1]
        return last_record

    def _extract(self, folder):
        """
        1. Download patient layout.
            -Is PLETH & ABP present in record?
        2. Download patient master header?
            -Is a segment longer than 10 minutes?
            -Does a segment have PLETH & ABP?
        3. 
        """
        valid_pleth = np.empty((0, 625))
        valid_abp = np.empty((0, 2))
        n_samples = 0

        mrn = folder.split('/')[1]
        layout = self._get_layout(folder)
        if layout and self._patient_has_pleth_abp(layout):
            master_header = self._get_master_header(folder)
            if master_header:
                segments = self._valid_segments(folder, master_header)
                print(f'Processing data for {folder}')
                for seg in segments:
                    pleth, abp = self._get_sigs(folder, seg)
                    if (len(pleth) > 0) & (len(abp) > 0):
                        sig_processor = SignalProcessor(pleth, abp, fs=125)
                        seg_pleth, seg_abp, seg_n_samples = sig_processor.run()
                        if seg_n_samples > 0:
                            valid_pleth = np.vstack([valid_pleth, seg_pleth])
                            valid_abp = np.vstack([valid_abp, seg_abp])
                            n_samples += seg_n_samples
        if n_samples > 0:
            append_patient(self._pleth_csv,
                           self._abp_csv,
                           mrn,
                           valid_pleth,
                           valid_abp,
                           n_samples)
            append_sample_count(self._data_profile_csv, mrn, n_samples)
        else:
            append_sample_count(self._data_profile_csv, mrn, 0)

        rmtree(f'physionet.org/files/mimic3wdb/1.0/{folder}', ignore_errors=True)
        return

    def _download(self, path):
        response = os.system(f'wget -q -r -np {path}')
        return response

    def _get_layout(self, folder):
        print(f'Getting layout file for {folder}')
        file = folder.split('/')[1] + '_layout'
        path = self._data_dir + folder + file
        response = self._download(path + '.hea')
        if response == 0:
            layout = rdheader(path)
            return layout
        return None

    def _patient_has_pleth_abp(self, layout):
        if ('PLETH' in layout.sig_name) & ('ABP' in layout.sig_name):
            return True
        return False

    def _get_master_header(self, folder):
        print(f'Getting master header file for {folder}')
        file = folder.split('/')[1]
        path = self._data_dir + folder + file
        response = self._download(path + '.hea')
        if response == 0:
            master_header = rdheader(path)
            return master_header
        return None

    def _valid_segments(self, folder, master_header):
        print(f'Getting valid segments for {folder}')
        seg_name = master_header.seg_name
        seg_len = master_header.seg_len

        segments = []
        for name, n_samples in zip(seg_name, seg_len):
            if (n_samples > 75000) & (name != '~'):
                path = self._data_dir + folder + name
                response = self._download(path + '.hea')
                if response == 0:
                    hea = rdheader(path)
                    if ('PLETH' in hea.sig_name) & ('ABP' in hea.sig_name):
                        segments.append(name)
        return segments

    def _get_sigs(self, folder, seg):
        path = self._data_dir + folder + seg
        response = self._download(path + '.dat')
        if response == 0:
            rec = rdrecord(path)
            signals = rec.sig_name
            pleth = rec.p_signal[:, signals.index('PLETH')].astype(np.float64)
            abp = rec.p_signal[:, signals.index('ABP')].astype(np.float64)
            return pleth, abp
        return None, None

    def _save_samples(self):
        return
