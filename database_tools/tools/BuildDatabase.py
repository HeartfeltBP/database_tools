import os
import json
import random
import numpy as np
import pandas as pd
from wfdb import rdheader, rdrecord
from .SignalProcessor import SignalProcessor


class BuildDatabase():
    def __init__(self,
                 records_path,
                 used_records_path,
                 samples_per_file,
                 max_samples,
                 data_dir='physionet.org/files/mimic3wdb/1.0/',
                 output_dir='data/mimic3/'):
        self._records_path = records_path
        self._used_records_path = used_records_path
        self._samples_per_file = samples_per_file
        self._max_samples = max_samples
        self._data_dir = data_dir
        self._output_dir = output_dir
    
    def run(self):
        total_samples = 0
        file_number = 0
        num_samples = 0
        sample_list = []

        for sample in self._sample_generator():
            sample_list.append(json.dumps(sample))
            total_samples += 1
            num_samples += 1

            if num_samples == self._samples_per_file:
                self._write(sample_list, file_number)
                file_number += 1
                num_samples = 0
                sample_list = []
                if total_samples >= self._max_samples:
                    break
        return

    def _get_used_records(self, path):
        used_records = set(pd.read_csv(path, index_col=['index'])['folder'])
        return used_records

    def _mrn_generator(self, path, used_records):
        records = set(pd.read_csv(path, names=['records'])['records'])
        records = list(records.difference(used_records))
        random.shuffle(records)
        for folder in records:
            yield folder

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

    def _sample_generator(self):
        used_records = self._get_used_records(self._used_records_path)
        for folder in self._mrn_generator(self._records_path, used_records):
            mrn = folder.split('/')[1]
            layout = self._get_layout(folder)
            if layout and self._patient_has_pleth_abp(layout):
                master_header = self._get_master_header(folder)
                if master_header:
                    segments = self._valid_segments(folder, master_header)
                    print(f'Processing data for {folder}')
                    sig_processor = SignalProcessor(segments, folder, mrn, self._data_dir, fs=125, window_size=5)
                    for sample in sig_processor.sample_generator():
                        yield sample
            used_records.add(folder)
            df = pd.DataFrame(used_records, columns=['folder'])
            df.index.names = ['index']
            df.to_csv(self._used_records_path)

    def _write(self, samples, file_number):
        file_name = f'mimic3_{str(file_number).zfill(8)}.json'
        print(f'Writing data to {file_name}')
        with open(self._output_dir + file_name, 'w') as f:
            for s in samples:
                json.dump(s, f)
        return
