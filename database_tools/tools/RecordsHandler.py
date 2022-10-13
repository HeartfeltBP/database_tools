import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm


class RecordsHandler():
    def __init__(self, data_dir):
        self._data_dir = data_dir

    def generate_records(self, split_strategy=(0.70, 0.15, 0.15), max_samples=1000):
        print('Loading data...')
        df = self._compile_lines(self._data_dir)
        ppg_set = np.array(df['ppg'].to_list())
        abp_set = np.array(df['abp'].to_list())

        print('Splitting data...')
        idx = np.random.permutation(len(df))
        i = int(len(idx) * split_strategy[0])
        j = int(len(idx) * split_strategy[1])

        idx_train = idx[0:i]
        idx_val = idx[i:i+j]
        idx_test = idx[i+j::]

        ppg_train = ppg_set[idx_train]
        ppg_val = ppg_set[idx_val]
        ppg_test = ppg_set[idx_test]
        
        abp_train = abp_set[idx_train]
        abp_val = abp_set[idx_val]
        abp_test = abp_set[idx_test]

        ppg_dict=dict(
            train=ppg_train,
            val=ppg_val,
            test=ppg_test
        )

        abp_dict=dict(
            train=abp_train,
            val=abp_val,
            test=abp_test,
        )

        print('Generating records...')
        output_dir = self._data_dir + 'records/'
        for split in ['train', 'val', 'test']:
            print(f'Starting {split} split...')
            ppg = ppg_dict[split]
            abp = abp_dict[split]

            file_number = 0
            num_samples = 0
            examples = []
            for ppg_win, abp_win in tqdm(zip(ppg, abp)):
                examples.append(
                    self._full_wave_window_example(
                        ppg=ppg_win,
                        abp=abp_win,
                    )
                )
                num_samples += 1
                if ((num_samples % max_samples) == 0) | (num_samples == len(ppg)):
                    self._write_record(examples, split, file_number, output_dir)
                    file_number += 1
                    examples = []
        print('Done!')
        return df

    def _compile_lines(self, path):
        frames = []
        for path in glob.glob(f'{path}lines/mimic3_*.jsonlines'):
            frames.append(pd.read_json(path, lines=True))
        df = pd.concat(frames, ignore_index=True)
        return df

    def _float_array_feature(self, value):
        """Returns a float_list from a float list."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def _full_wave_window_example(self, ppg, abp):
        feature = {
            'ppg' : self._float_array_feature(ppg),
            'abp' : self._float_array_feature(abp),
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        return example

    def _write_record(self, examples, split, file_number, path):
        file_name = path + split + f'/mimic3_{str(file_number).zfill(7)}.tfrecords'
        with tf.io.TFRecordWriter(file_name) as w:
            for tf_example in examples:
                w.write(tf_example.SerializeToString())

    def read_records(self):
        return
