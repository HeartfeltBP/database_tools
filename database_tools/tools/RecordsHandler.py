import glob
import time
import numpy as np
import pandas as pd
import pickle as pkl
import tensorflow as tf
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class RecordsHandler():
    def __init__(self, data_dir):
        self._data_dir = data_dir

    def generate_records(self, split_strategy=(0.70, 0.15, 0.15), max_samples=10000):
        """
        Generates TFRecords files from JSONLINES files.

        Args:
            split_strategy (tuple, optional): Data % for train, val, test. Defaults to (0.70, 0.15, 0.15).
            max_samples (int, optional): Max samples per TFRecords file. Defaults to 10000.
            normalize (str, optional): Normalization method. Either 'MinMax' or 'Standard'. Defaults to 'MinMax'.
        """
        print('Loading data...')
        df = self._compile_lines(self._data_dir)

        # Determine indices for split
        idx = [i for i in range(0, len(df))]
        test_size = split_strategy[1] + split_strategy[2]
        idx_train, idx_val = train_test_split(
            idx,
            train_size=split_strategy[0],
            test_size=test_size
        )
        idx_val, idx_test = train_test_split(
            idx_val,
            train_size=round(split_strategy[1] / test_size, 5),
            test_size=round(split_strategy[2] / test_size, 5)
        )

        data_unscaled = {
            'ppg': np.array(df['ppg'].to_list()),
            'abp': np.array(df['abp'].to_list()),
        }
        data_unscaled['vpg'] = np.gradient(data_unscaled['ppg'], axis=1)  # 1st derivative of ppg
        data_unscaled['apg'] = np.gradient(data_unscaled['vpg'], axis=1)  # 2nd derivative of vpg

        labels = ['ppg', 'vpg', 'apg', 'abp']

        print(f'Scaling and splitting data...')
        data = {}
        scalers = {}
        for key in labels:
            std_scaler = StandardScaler()
            temp = std_scaler.fit_transform(data_unscaled[key])

            train = temp[idx_train]
            val = temp[idx_val]
            test = temp[idx_test]
            data[key] = dict(
                train=train,
                val=val,
                test=test,
            )
            scalers[key] = std_scaler

        r = time.time_ns()
        with open(f'{self._data_dir}scalers_{r}.pkl', 'wb') as f:
            pkl.dump(scalers, f)

        print('Generating records...')
        output_dir = self._data_dir + f'records/'
        for i, split in enumerate(['train', 'val', 'test']):
            print(f'Starting {split} split...')
            split_data = {f'{k}': data[k][split] for k in data.keys()}

            file_number = 0
            num_samples = 0
            examples = []

            total = len(idx) * split_strategy[i]
            for win in tqdm(zip(*split_data.values()), total=int(total)):
                examples.append(
                    self._full_wave_window_example(
                        win=win,
                        labels=labels,
                    )
                )
                num_samples += 1
                if ((num_samples % max_samples) == 0) | (num_samples == total):
                    self._write_record(examples, split, file_number, output_dir)
                    file_number += 1
                    examples = []
        print('Done!')
        return

    def _compile_lines(self, path):
        frames = []
        for path in glob.glob(f'{path}lines/mimic3_*.jsonlines'):
            frames.append(pd.read_json(path, lines=True))
        df = pd.concat(frames, ignore_index=True)
        return df

    def _float_array_feature(self, value):
        """Returns a float_list from a float list."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def _full_wave_window_example(self, win, labels):
        feature = {f'{l}': self._float_array_feature(win[i]) for i, l in enumerate(labels)}
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        return example

    def _write_record(self, examples, split, file_number, path):
        file_name = path + split + f'/mimic3_{str(file_number).zfill(7)}.tfrecords'
        with tf.io.TFRecordWriter(file_name) as w:
            for tf_example in examples:
                w.write(tf_example.SerializeToString())

    def read_records(self, splits, labels, n_cores, AUTOTUNE):
        data_splits = {}
        for s in splits:
            print(f'Reading {s} split.')
            filenames = [file for file in glob.glob(f'{self._data_dir}records/{s}/*.tfrecords')]
            dataset = tf.data.TFRecordDataset(
                filenames=filenames,
                compression_type=None,
                buffer_size=100000000,
                num_parallel_reads=n_cores
            )
            if labels == ['ppg', 'vpg', 'apg', 'abp']:
                data_splits[s] = dataset.map(self._full_wave_parse_window_function3, num_parallel_calls=AUTOTUNE)
            elif labels == ['ppg', 'vpg', 'abp']:
                data_splits[s] = dataset.map(self._full_wave_parse_window_function2, num_parallel_calls=AUTOTUNE)
            elif labels == ['ppg', 'abp']:
                data_splits[s] = dataset.map(self._full_wave_parse_window_function1, num_parallel_calls=AUTOTUNE)
            else:
                raise ValueError(f'Invalid labels: {labels}')
        return data_splits

    def _full_wave_parse_window_function3(self, example_proto):
        print('test')
        features = tf.io.parse_single_example(
            example_proto, 
            features = {
                'ppg': tf.io.FixedLenFeature([256], tf.float32),
                'vpg': tf.io.FixedLenFeature([256], tf.float32),
                'apg': tf.io.FixedLenFeature([256], tf.float32),
                'abp': tf.io.FixedLenFeature([256], tf.float32),
            }
        )
        ppg = tf.reshape(features['ppg'], (256, 1))
        vpg = tf.reshape(features['vpg'], (256, 1))
        apg = tf.reshape(features['apg'], (256, 1))
        inputs = dict(ppg=ppg, vpg=vpg, apg=apg)
        label = tf.reshape(features['abp'], (256, 1))
        return inputs, label

    def _full_wave_parse_window_function2(self, example_proto):
        features = tf.io.parse_single_example(
            example_proto, 
            features = {
                'ppg': tf.io.FixedLenFeature([256], tf.float32),
                'vpg': tf.io.FixedLenFeature([256], tf.float32),
                'abp': tf.io.FixedLenFeature([256], tf.float32),
            }
        )
        ppg = tf.reshape(features['ppg'], (256, 1))
        vpg = tf.reshape(features['vpg'], (256, 1))
        inputs = dict(ppg=ppg, vpg=vpg)
        label = tf.reshape(features['abp'], (256, 1))
        return inputs, label

    def _full_wave_parse_window_function1(self, example_proto):
        features = tf.io.parse_single_example(
            example_proto, 
            features = {
                'ppg': tf.io.FixedLenFeature([256], tf.float32),
                'abp': tf.io.FixedLenFeature([256], tf.float32),
            }
        )
        ppg = tf.reshape(features['ppg'], (256, 1))
        inputs = dict(ppg=ppg)
        label = tf.reshape(features['abp'], (256, 1))
        return inputs, label
