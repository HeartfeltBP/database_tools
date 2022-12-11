import json
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from database_tools.preprocessing import SignalProcessor


class BuildDatabase():
    def __init__(self,
                 output_dir,
                 config,
                 win_len=1024,
                 fs=125,
                 samples_per_file=6000,
                 max_samples=5000,
                 data_dir='physionet.org/files/mimic3wdb/1.0/'):
        self._output_dir = output_dir
        self._config = config
        self._win_len = win_len
        self._fs = fs
        self._samples_per_file = samples_per_file
        self._max_samples = max_samples
        self._data_dir = data_dir

    def run(self):
        valid_segs_path = self._output_dir + 'valid_segs.csv'
        used_segs_path = self._output_dir + 'used_segs.pkl'
        valid_segs = self._get_valid_segs(valid_segs_path, used_segs_path)

        sample_gen = SignalProcessor(
            files=valid_segs,
            output_dir=self._output_dir,
            win_len=self._win_len,
            fs=self._fs,
        )

        samples = ''
        try:
            n_samples = (np.max([int(i.split('/')[-1][7:14]) for i in glob(self._output_dir + 'mimic3/lines/' + '*.jsonlines')]) + 1) * self._samples_per_file
        except:
            n_samples = 0
        print('Starting sample generator...')
        for ppg, abp in tqdm(sample_gen.run(config=self._config), total=self._max_samples):
            samples += json.dumps(dict(ppg=ppg.tolist(), abp=abp.tolist())) + '\n'
            n_samples += 1

            if (n_samples % self._samples_per_file) == 0:
                file_number = str(int(n_samples / self._samples_per_file) - 1).zfill(7)
                outfile = self._output_dir + f'mimic3/lines/mimic3_{file_number}.jsonlines'
                self._write_to_jsonlines(samples, outfile)
                samples = ''
                if n_samples >= self._max_samples:
                    break

        print('Saving stats...')
        sample_gen.save_stats(self._output_dir + 'mimic3_stats.csv')
        print('Done!')
        return

    def _get_valid_segs(self, valid_path, used_path):
        df_valid = pd.read_csv(valid_path, names=['url'])
        try:
            df_used = pd.read_csv(used_path, names=['url'])
            all_valid = set(df_valid['url'])
            used = set(df_used['url'])
            valid = list(all_valid.difference(used))
            return pd.Series(valid)
        except FileNotFoundError:
            return df_valid['url']

    def _write_to_jsonlines(self, output, outfile):
        with open(outfile, 'w') as f:
            f.write(output)
