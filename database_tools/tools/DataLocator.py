import os
import pandas as pd
from datetime import date
from tqdm import tqdm
from shutil import rmtree
from wfdb import rdheader


class DataLocator():
    def __init__(self,
                 data_dir='physionet.org/files/mimic3wdb/1.0/'):
        self._data_dir = data_dir

    def run(self):
        records_path = self._data_dir + 'RECORDS-adults'
        out_path = self._build_empty_database(records_path)

        valid_segs = []
        i = 0
        for folder in tqdm(self._mrn_generator(out_path + 'RECORDS-adults'), total=59344):
            i += 1
            layout = self._get_layout(folder)
            if not layout:
                continue
            if not self._patient_has_pleth_abp(layout):
                continue
            master_header = self._get_master_header(folder)
            if not master_header:
                continue
            valid_segs += self._valid_segments(folder, master_header)
            if (i % 100) == 0:
                rmtree(self._data_dir, ignore_errors=True)

        pd.DataFrame(valid_segs).to_csv(out_path + 'valid_segs.csv', index=False)
        return valid_segs

    def _build_empty_database(self, records_path):
        path = f'data-{str(date.today())}/'
        os.mkdir(path)
        os.system(f'wget -q -np https://{records_path} -P {path}')
        return path

    def _mrn_generator(self, path):
        records = set(pd.read_csv(path, names=['records'])['records'])
        for folder in records:
            yield folder

    def _download(self, path):
        response = os.system(f'wget -q -r -np https://{path}')
        return response

    def _get_layout(self, folder):
        file = folder.split('/')[1] + '_layout'
        path = self._data_dir + folder + file
        response = self._download(path + '.hea')
        if response == 0:
            layout = rdheader(path)
            return layout
        return False

    def _patient_has_pleth_abp(self, layout):
        if (layout is None) | (layout.sig_name is None):
            return False
        elif ('PLETH' in layout.sig_name) & ('ABP' in layout.sig_name):
            return True
        return False

    def _get_master_header(self, folder):
        file = folder.split('/')[1]
        path = self._data_dir + folder + file
        response = self._download(path + '.hea')
        if response == 0:
            master_header = rdheader(path)
            return master_header
        return False

    def _valid_segments(self, folder, master_header):
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
                        segments.append('https://' + path)
        return segments
