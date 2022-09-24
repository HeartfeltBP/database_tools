import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from database_tools.preprocessing import SignalProcessor


class BuildDatabase():
    def __init__(self,
                 output_dir,
                 config,
                 win_len=1024,
                 fs=125,
                 data_dir='physionet.org/files/mimic3wdb/1.0/'):
        self._output_dir = output_dir
        self._config = config
        self._win_len = win_len
        self._fs = fs
        self._data_dir = data_dir

    def run(self):
        valid_segs_path = self._output_dir + 'valid_segs.csv'
        valid_segs = self._get_valid_segs(valid_segs_path)

        sample_gen = SignalProcessor(
            files=valid_segs['url'],
            win_len=self._win_len,
            fs=self._fs,
        )

        windows = []
        i = 0
        print('Starting sample generator...')
        for win in tqdm(sample_gen.run(config=self._config), total=1000):
            windows.append(win)
            i += 1
            if i == 1000:
                break
        n_excluded, sim, snr, hr, abp_max, abp_min = sample_gen.get_stats()
        return np.array(windows), n_excluded, sim, snr, hr, abp_max, abp_min

    def _get_valid_segs(self, path):
        df = pd.read_csv(path, names=['url'])
        return df

    def _save_metrics(self, n_excluded, sim, snr, hr, abp_max, abp_min):
        return

    def _write_to_jsonlines(self, data, path):
        return
