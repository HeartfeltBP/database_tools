import glob
import numpy as np
import pickle as pkl
from tqdm import tqdm
import tensorflow as tf
from time import time_ns
from typing import Tuple, List
from sklearn.model_selection import train_test_split
from database_tools.tools.dataset import Dataset

def generate_records(
    ds: Dataset,
    data_dir: str,
    split_strategy: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    samples_per_file: int = 10000,
    scaler_path: str = None,
) -> Tuple[dict, dict]:
    """Generates TFRecords files from dataset.

    Args:
        ds (Dataset): Dataset object consisting of ppg and abp data from JSONLINES files.
                      Other data includes vpg, and apg data derived from ppg.

        split_strategy (Tuple[float, float, float], optional): Train, val, test split percentages.
            If not provided the entire dataset will be treated as test data.

        samples_per_file (int, optional): Max samples per tfrecords file. Defaults to 10000.

        scaler_path (path, optional): Path to an existing scaler. If not provided a scaler will be
            created based on given dataset.
    """
    print('Splitting data...')
    if split_strategy is None:
        idx = dict(train=[], val=[], test=[i for i in range(0, ds.ppg.shape[0])])
    else:
        idx = get_split_idx(n=ds.ppg.shape[0], split_strategy=split_strategy)
    data_unscaled = split_data(ds, idx)

    print('Scaling data...')
    if scaler_path is not None:
        with open(scaler_path, 'rb') as f:
            scaler, scaler_split_idx = pkl.load(f)
    else:
        scaler = None
    data_scaled, scaler_dict = scale_data(data_unscaled, scaler)

    print('Generating TFRecords...')
    write_records(
        data=data_scaled,
        data_dir=data_dir,
        samples_per_file=samples_per_file,
    )
    if scaler_path is None:
        with open(f'{data_dir}records_info_{time_ns()}.pkl', 'wb') as f:
            pkl.dump([scaler_dict, idx], f)
    return (data_unscaled, data_scaled, scaler_dict)

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

def split_data(ds: Dataset, idx) -> dict:
    data = {}
    for key in ['ppg', 'vpg', 'apg', 'abp']:
        tmp = ds.__getattribute__(key)
        data[key] = dict(
            train=tmp[idx['train']],
            val=tmp[idx['val']],
            test=tmp[idx['test']],
        )
    return data

def scale_data(data_unscaled: dict, scaler: dict) -> Tuple[dict, dict]:
    data_scaled = {'ppg': {}, 'vpg': {}, 'apg': {}, 'abp': {}}
    scaler_dict = {}
    for key in ['ppg', 'vpg', 'apg']:
        if scaler is None:
            min_ = np.min(data_unscaled[key]['train'])
            max_ = np.max(data_unscaled[key]['train'])
        else:
            min_ = scaler[key][0]
            max_ = scaler[key][1]
        scaler_dict[key] = [min_, max_]

        for split in ['train', 'val', 'test']:
            tmp = data_unscaled[key][split]
            tmp_scaled = np.divide(tmp - min_, max_ - min_)
            data_scaled[key][split] = tmp_scaled
    data_scaled['abp']['train'] = data_unscaled['abp']['train']
    data_scaled['abp']['val'] = data_unscaled['abp']['val']
    data_scaled['abp']['test'] = data_unscaled['abp']['test']
    return (data_scaled, scaler_dict)

def write_records(data: dict, data_dir: str, samples_per_file: int) -> None:
    records_dir = data_dir + 'data/records/'
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
