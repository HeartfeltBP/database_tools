import os
import numpy as np
import pandas as pd
from wfdb import rdheader, rdrecord
from shutil import rmtree
from database_tools.utils.compile import append_sample_count, write_dataset
from .SignalProcessor import SignalProcessor


class BuildDatabase():
    def __init__(self,
                 records_path,
                 data_profile_csv,
                 max_records,
                 max_file_size,  # In MB
                 data_dir='physionet.org/files/mimic3wdb/1.0/',
                 output_dir='data/mimic3/'):
        self._records_path = records_path
        self._data_profile_csv = data_profile_csv
        self._max_records = max_records
        self._max_file_size = max_file_size * (10 ** 6)
        self._data_dir = data_dir
        self._output_dir = output_dir

    def run(self):
        # Load records.
        records = pd.read_csv(self._records_path, names=['patient_dir'])

        # Get last record (MRN) added to data profile.
        last_record = str(self._get_last_record(self._data_profile_csv))

        start = 0  # Start idx is 0 unless database is partially built.
        if last_record != 'test':
            start = records.index[records['patient_dir'].str.contains(last_record)].tolist()[0] + 1

        num_processed = 0
        file_number = 0
        for folder in records['patient_dir'][start::]:
            if num_processed == self._max_records:
                break

            file_name = self._output_dir + f'mimic3_{file_number}.tfrecords'
            self._extract(folder, file_name=file_number)

            num_processed += 1
            if os.path.exists(file_name):
                if os.path.getsize(file_name) > self._max_file_size:
                    file_number += 1
        return

    def _get_last_record(self, path):
        df = pd.read_csv(path, index_col='index')
        last_record = df['mrn'].iloc[-1]
        return last_record

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

    def _extract(self, folder, file_name):
        """
        1. Download patient layout.
            -Is PLETH & ABP present in record?
        2. Download patient master header?
            -Is a segment longer than 10 minutes?
            -Does a segment have PLETH & ABP?
        3. 
        """
        valid_samples = []
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
                        sig_processor = SignalProcessor(pleth, abp, mrn, fs=125)
                        seg_valid_samples, seg_n_samples = sig_processor.run()
                        if seg_n_samples > 0:
                            valid_samples += seg_valid_samples
                            n_samples += seg_n_samples
        if n_samples > 0:
            write_dataset(file_name=file_name,
                          examples=valid_samples)
            append_sample_count(self._data_profile_csv, mrn, n_samples)
        else:
            append_sample_count(self._data_profile_csv, mrn, 0)

        rmtree(f'physionet.org/files/mimic3wdb/1.0/{folder}', ignore_errors=True)
        return
