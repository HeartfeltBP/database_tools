import json
import pandas as pd
from tqdm import tqdm
from database_tools.preprocessing import SignalProcessor


class BuildDatabase():
    def __init__(
        self,
        data_dir,
        config,
        win_len=256,
        fs=125,
        samples_per_file=2500,
        samples_per_patient=500,
        max_samples=300000,
    ) -> None:
        self._data_dir = data_dir
        self._config = config
        self._win_len = win_len
        self._fs = fs
        self._samples_per_file = samples_per_file
        self._samples_per_patient = samples_per_patient
        self._max_samples = max_samples

    def run(self):
        valid_df = self._get_valid_segs(self._data_dir + 'valid_segs.csv')
        partner = self._data_dir.split('-')[0]

        # Initialize SignalProcessor()
        sample_gen = SignalProcessor(
            partner=partner,
            valid_df=valid_df,
            win_len=self._win_len,
            fs=self._fs,
            samples_per_patient=self._samples_per_patient,
        )

        print('Starting sample generator...')

        samples, n_samples = '', 0
        for ppg, abp in tqdm(sample_gen.run(config=self._config), total=self._max_samples):
            samples += json.dumps(dict(ppg=ppg.tolist(), abp=abp.tolist())) + '\n'
            n_samples += 1

            if (n_samples % self._samples_per_file) == 0:
                fn = str(int(n_samples / self._samples_per_file) - 1).zfill(3)  # file name
                outfile = self._output_dir + f"data/lines/{self._config['partner']}_{fn}.jsonlines"
                self._write_to_jsonlines(samples, outfile)
                samples = ''
                if n_samples >= self._max_samples:
                    break

        print('Saving stats...')
        sample_gen.save_stats(self._data_dir + 'mimic3_stats.csv')
        print('Done!')
        return

    def _get_valid_segs(self, valid_path):
        valid_df = pd.read_csv(valid_path)
        return valid_df

    def _write_to_jsonlines(self, output, outfile):
        with open(outfile, 'w') as f:
            f.write(output)
