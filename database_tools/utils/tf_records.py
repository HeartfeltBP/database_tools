import numpy as np
import tensorflow as tf

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _float_array_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def window_example(sig, sbp, dbp, mrn):
    feature = {
        'sig': _float_array_feature(sig),
        'sbp': _float_feature(sbp),
        'dbp': _float_feature(dbp),
        'mrn': _int64_feature(mrn),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def _parse_window_function(example_proto):
    window_feature_description = {
        'sig': tf.io.FixedLenFeature([], tf.float64),
        'sbp': tf.io.FixedLenFeature([], tf.float64),
        'dbp': tf.io.FixedLenFeature([], tf.float64),
        'mrn': tf.io.FixedLenFeature([], tf.int64)
    }
    return tf.io.parse_single_example(example_proto, window_feature_description)

def read_dataset(path, n_cores=1):
    dataset = tf.data.TFRecordDataset(filenames=[],
                                      compression_type=None,
                                      buffer_size=None,
                                      num_parallel_reads=n_cores)
    parsed_dataset = dataset.map(_parse_window_function)
    return parsed_dataset
