import glob
import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm
from time import time_ns
import tensorflow as tf
from sklearn.model_selection import train_test_split

import logging
from typing import Tuple, List
from dataclasses import dataclass, InitVar, field

logging.basicConfig(level=logging.INFO)


@dataclass
class Dataset:

    data_dir: InitVar[str]
    ppg: np.ndarray = field(init=False)
    vpg: np.ndarray = field(init=False)
    apg: np.ndarray = field(init=False)
    abp: np.ndarray = field(init=False)

    _nframes: int = field(init=False)

    def __post_init__(self, data_dir: str) -> None:
        frames = [pd.read_json(file, lines=True) for file in glob.glob(f'{data_dir}lines/*.jsonlines')]
        object.__setattr__(self, '_nframes', len(frames))
        data = pd.concat(frames, ignore_index=True)
        object.__setattr__(self, 'ppg', np.array(data['ppg'].to_list()))
        vpg = np.gradient(self.ppg, axis=1)  # 1st derivative of ppg
        apg = np.gradient(vpg, axis=1) # 2nd derivative of vpg
        object.__setattr__(self, 'vpg', vpg)
        object.__setattr__(self, 'apg', apg)
        object.__setattr__(self, 'abp', np.array(data['abp'].to_list()))
        self.info

    @property
    def info(self):
        logging.info(f'Data was extracted from {self._nframes} JSONLINES files.')
        logging.info(f'The total number of windows is {self.ppg.shape[0]}.')


def generate_records(
    ds: Dataset,
    data_dir: str,
    split_strategy: Tuple[float, float, float],
    samples_per_file: int = 10000,
) -> Tuple[dict, dict]:
    """Generates TFRecords files from dataset.

    Args:
        ds (Dataset): Dataset object consisting of ppg and abp data from JSONLINES files.
                      Other data includes vpg, and apg data derived from ppg.
        split_strategy (Tuple[float, float, float]): Train, val, test split percentages.
        samples_per_file (int, optional): Max samples per tfrecords file. Defaults to 10000.
    """
    print('Determing split indices...')
    idx = get_split_idx(n=ds.ppg.shape[0], split_strategy=split_strategy)

    print('Scaling and splitting data...')
    data, scalers = scale_data(ds, idx)

    with open(f'{data_dir}min_max_scaler_{time_ns()}.pkl', 'wb') as f:
        pkl.dump(scalers, f)

    print('Generating TFRecords...')
    write_records(
        data=data,
        data_dir=data_dir,
        samples_per_file=samples_per_file,
    )
    return (data, scalers)

def get_split_idx(n: int, split_strategy: Tuple[float, float, float]) -> dict:
    idx = [i for i in range(0, n)]
    tmp = split_strategy[1] + split_strategy[2]  # val + test size
    idx_train, idx_val = train_test_split(
        idx,
        train_size=split_strategy[0],
        test_size=tmp
    )
    idx_val, idx_test = train_test_split(
        idx_val,
        train_size=round(split_strategy[1] / tmp, 2),
        test_size=round(split_strategy[2] / tmp, 2)
    )
    return dict(train=idx_train, val=idx_val, test=idx_test)

def scale_data(ds: Dataset, idx: dict) -> Tuple[dict, dict]:
    data, scalers = {}, {}
    for key in ['ppg', 'vpg', 'apg', 'abp']:
        tmp = ds.__getattribute__(key)
        min_ = np.min(tmp)
        max_ = np.max(tmp)

        tmp_scaled = np.divide(tmp - min_, max_ - min_)

        scalers[key] = [min_, max_]
        data[key] = dict(
            train=tmp_scaled[idx['train']],
            val=tmp_scaled[idx['val']],
            test=tmp_scaled[idx['test']],
        )
    return (data, scalers)

def write_records(data: dict, data_dir: str, samples_per_file: int) -> None:
    records_dir = data_dir + 'records/'
    for split in ['train', 'val', 'test']:
        print(f'Starting {split} split...')
        split_data = {f'{sig}': data[sig][split] for sig in data.keys()}

        file_number = 0
        num_samples = 0
        examples = []
        total = split_data['ppg'].shape[0]
        for win in tqdm(zip(*split_data.values()), total=int(total)):
            examples.append(
                full_wave_window_example(
                    win=win,
                    labels=['ppg', 'vpg', 'apg', 'abp'],
                )
            )
            num_samples += 1
            if ((num_samples % samples_per_file) == 0) | (num_samples == total):
                file_name = records_dir + split + f'/mimic3_{str(file_number).zfill(3)}.tfrecords'
                with tf.io.TFRecordWriter(file_name) as w:
                    for tf_example in examples:
                        w.write(tf_example.SerializeToString())
                file_number += 1
                examples = []

def float_array_feature(value):
    """Returns a float_list from a float list."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def full_wave_window_example(win, labels):
    feature = {f'{l}': float_array_feature(win[i]) for i, l in enumerate(labels)}
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example

def read_records(data_dir: str, n_cores: int = 10, n_parallel: int = tf.data.AUTOTUNE) -> dict:
    data = {}
    for split in ['train', 'val', 'test']:
        print(f'Reading {split} split.')
        filenames = [file for file in glob.glob(f'{data_dir}records/{split}/*.tfrecords')]
        dataset = tf.data.TFRecordDataset(
            filenames=filenames,
            compression_type=None,
            buffer_size=100000000,
            num_parallel_reads=n_cores
        )
        data[split] = dataset.map(full_wave_parse_window_function, num_parallel_calls=n_parallel)
    return data

def full_wave_parse_window_function(example_proto):
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

def rescale_data(sig: np.ndarray, scaler: List[int]) -> np.ndarray:
    """Rescale sample(s) with min max scaler provided in the
       form of List[min, max].

    Args:
        sig (np.ndarray): Array of samples.
        scaler (dict): List of min and max value (min-max scaler).

    Returns:
        np.ndarray: Scaled sample(s).
    """
    sig_scaled = np.multiply(sig, scaler[1] - scaler[0]) + scaler[0]
    return sig_scaled
