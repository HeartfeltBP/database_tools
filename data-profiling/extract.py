from operator import index
import os
import numpy as np
import pandas as pd
from preprocess import SignalProcessor
from wfdb import rdheader, rdrecord
from shutil import rmtree
from sklearn.utils import indexable
from tqdm.notebook import tqdm


class ExtractData():
    def __init__(self, records_path, last_record, max_records):
        self._records_path = records_path
        self._last_record = last_record
        self._max_records = max_records
        self._data_dir = 'physionet.org/files/mimic3wdb/1.0/'

    def run(self):
        records = pd.read_csv(self._records_path, names=['patient dir'])
        for folder in tqdm(records['patient dir']):
            self._extract(folder) 
        return

    def _extract(self, folder):
        """
        1. Download patient layout.
            -Is PLETH & ABP present in record?
        2. Download patient master header?
            -Is a segment longer than 10 minutes?
            -Does a segment have PLETH & ABP?
        3. 
        """
        layout = self._get_layout(folder)
        if layout and self._patient_has_pleth_abp(layout):
            master_header = self._get_master_header(folder)
            if master_header:
                segments = self._valid_segments(folder, master_header)
                for seg in segments:
                    pleth, abp = self._get_sigs(folder, seg)
                    sig_processor = SignalProcessor(pleth, abp, fs=125)
                    # ...
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
