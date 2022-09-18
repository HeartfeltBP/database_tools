import sys
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm.notebook import tqdm
from database_tools.tools import Normalizer

class CompileDatabase():
    def __init__(self, path):
        self._path = path

    def run(self):
        frames = []
        for path in glob.glob(f'{self._path}mimic3_*.jsonlines'):
            frames.append(pd.read_json(path, lines=True))
        df = pd.concat(frames, ignore_index=True)
        return df


class GenerateTFRecords():
    def __init__(self,
                 data_dir,
                 output_dir,
                 method,
                 split_strategy=(.7, .15, .15),
                 max_samples=1000):
        self._data_dir = data_dir
        self._output_dir = output_dir
        self._method = method
        self._split_strategy = split_strategy
        self._max_samples = max_samples

    def run(self):
        """
        Methods
            'Full Waves' : x = full pleth wave, y = full abp wave
        """
        print('Loading data.')
        df = CompileDatabase(path=self._data_dir).run()
        self._num_samples = len(df)

        pleth = np.array(df['pleth'].to_list())
        abp = np.array(df['abp'].to_list())

        print('Scaling data.')
        scaler = Normalizer()
        pleth = scaler.fit_transform(pleth)

        print('Splitting data.')
        idx = np.random.permutation(self._num_samples)
        i = int(len(idx) * self._split_strategy[0])
        j = int(len(idx) * self._split_strategy[1])

        idx_train = idx[0:i]
        idx_val = idx[i:i+j]
        idx_test = idx[i+j::]

        pleth_train = pleth[idx_train]
        pleth_val = pleth[idx_val]
        pleth_test = pleth[idx_test]
        
        abp_train = abp[idx_train]
        abp_val = abp[idx_val]
        abp_test = abp[idx_test]

        if self._method == 'Full Waves':
            self._full_wave(
                pleth_dict=dict(
                    train=pleth_train,
                    val=pleth_val,
                    test=pleth_test,
                ),
                abp_dict=dict(
                    train=abp_train,
                    val=abp_val,
                    test=abp_test,
                )
            )
        else:
            raise ValueError(f'Invalid method {self._method}')
        return

    def _float_array_feature(self, value):
        """Returns a float_list from a float list."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def _full_waves_window_example(self, pleth, abp):
        feature = {
            'pleth': self._float_array_feature(pleth),
            'abp'  : self._float_array_feature(abp),
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        return example

    def _full_wave(self, pleth_dict, abp_dict):
        for split in ['train', 'val', 'test']:
            pleth = pleth_dict[split]
            abp = abp_dict[split]

            file_number = 0
            num_samples = 0
            examples = []
            sys.stdout.write('\r' + f'Starting {split} split.')
            for pleth_win, abp_win in tqdm(zip(pleth, abp)):
                examples.append(self._full_waves_window_example(
                    pleth=pleth_win,
                    abp=abp_win,
                ))
                num_samples += 1
                if ((num_samples / self._max_samples) == 1.0) | (num_samples == len(pleth)):
                    self._write_record(examples, split, file_number, self._output_dir)
                    file_number += 1
                    examples = []

    def _write_record(self, examples, split, file_number, path):
        file_name = path + split + f'/mimic3_{str(file_number).zfill(8)}.tfrecords'
        print('writing')
        with tf.io.TFRecordWriter(file_name) as w:
            for tf_example in examples:
                w.write(tf_example.SerializeToString())


class ReadTFRecords():
    def __init__(self, data_dir, n_cores):
        self._data_dir = data_dir
        self._n_cores = n_cores

    def run(self):
        data_splits = {}
        for split in ['train', 'val', 'test']:
            print(f'Reading {split} split.')
            filenames = [file for file in glob.glob(f'{self._data_dir}{split}/*.tfrecords')]
            dataset = tf.data.TFRecordDataset(
                filenames=filenames,
                compression_type=None,
                buffer_size=None,
                num_parallel_reads=self._n_cores
            )
            data_splits[split] = dataset.map(self._full_waves_parse_window_function)
        return data_splits

    def _full_waves_parse_window_function(self, example_proto):
        window_feature_description = {
            'pleth': tf.io.FixedLenFeature([625], tf.float32),
            'abp': tf.io.FixedLenFeature([625], tf.float32),
        }
        return tf.io.parse_single_example(example_proto, window_feature_description)
