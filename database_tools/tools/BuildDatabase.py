import json
import pandas as pd
from tqdm import tqdm
from database_tools.preprocessing import SignalProcessor


class BuildDatabase():
    def __init__(self,
                 output_dir,
                 config,
                 win_len=1024,
                 fs=125,
                 samples_per_file=6000,
                 samples_per_patient=6000,
                 max_samples=5000,
                 data_dir='physionet.org/files/mimic3wdb/1.0/'):
        self._output_dir = output_dir
        self._config = config
        self._win_len = win_len
        self._fs = fs
        self._samples_per_file = samples_per_file
        self._samples_per_patient = samples_per_patient
        self._max_samples = max_samples
        self._data_dir = data_dir

    def run(self):
        valid_segs_path = self._output_dir + 'valid_segs.csv'
        valid_segs = self._get_valid_segs(valid_segs_path)

        sample_gen = SignalProcessor(
            files=valid_segs['url'],
            samples_per_patient=self._samples_per_patient,
            win_len=self._win_len,
            fs=self._fs,
        )

        samples = ''
        n_samples = 0
        print('Starting sample generator...')
        for ppg, abp in tqdm(sample_gen.run(config=self._config), total=self._max_samples):
            samples += json.dumps(dict(ppg=ppg.tolist(), abp=abp.tolist())) + '\n'
            n_samples += 1

            if (n_samples % self._samples_per_file) == 0:
                file_number = str(int(n_samples / self._samples_per_file) - 1).zfill(7)
                outfile = self._output_dir + f'mimic3/mimic3_{file_number}.jsonlines'
                self._write_to_jsonlines(samples, outfile)
                samples = ''
                if n_samples >= self._max_samples:
                    break

        print('Saving stats...')
        sample_gen.save_stats(self._output_dir + 'mimic3_stats.csv')
        print('Done!')
        return

    def _get_valid_segs(self, path):
        df = pd.read_csv(path, names=['url'])
        return df

    def _write_to_jsonlines(self, output, outfile):
        print(f'Writing to {outfile}')
        with open(outfile, 'w') as f:
            f.write(output)
