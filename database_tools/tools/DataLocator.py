import os
import pandas as pd
from tqdm import tqdm
from shutil import rmtree
from wfdb import rdheader


class MimicDataLocator():
    def __init__(
        self,
        data_dir='/path/to/data/dir/',
        mimic3_dir='physionet.org/files/mimic3wdb/1.0/',
    ) -> None:
        self._data_dir = data_dir
        self._mimic3_dir = mimic3_dir

    def run(self):
        records_path = self._mimic3_dir + 'RECORDS-adults'
        os.system(f'wget -q -np https://{records_path} -P {self._data_dir}')

        patient_ids, valid_segs = [], []
        for i, folder in tqdm(enumerate(self._mrn_generator(self._data_dir + 'RECORDS-adults')), total=59344):
            layout = self._get_layout(folder)
            if not layout:
                continue
            if not self._patient_has_pleth_abp(layout):
                continue
            master_header = self._get_master_header(folder)
            if not master_header:
                continue
            temp = self._valid_segments(folder, master_header)
            patient_ids += [folder.split('/')[1]] * len(temp)
            valid_segs += temp
            if (i % 100) == 0:
                rmtree(self._mimic3_dir, ignore_errors=True)

        df = pd.DataFrame(dict(id=patient_ids, url=valid_segs))
        df.to_csv(self._data_dir + 'valid_segs.csv', index=False)
        return

    def _mrn_generator(self, path):
        records = set(pd.read_csv(path, names=['records'])['records'])
        for folder in records:
            yield folder

    def _download(self, path):
        response = os.system(f'wget -q -r -np https://{path}')
        return response

    def _get_layout(self, folder):
        file = folder.split('/')[1] + '_layout'
        path = self._mimic3_dir + folder + file
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
        path = self._mimic3_dir + folder + file
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
                path = self._mimic3_dir + folder + name
                response = self._download(path + '.hea')
                if response == 0:
                    hea = rdheader(path)
                    if ('PLETH' in hea.sig_name) & ('ABP' in hea.sig_name):
                        segments.append('https://' + path)
        return segments
