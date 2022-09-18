import glob
import pandas as pd
# import tensorflow as tf
# from typing import List


class CompileDatabase():
    def __init__(self, path):
        self._path = path

    def run(self):
        frames = []
        for path in glob.glob(f'{self._path}mimic3_*.jsonlines'):
            frames.append(pd.read_json(path, lines=True))
        df = pd.concat(frames)
        return df


# def _float_array_feature(value):
#     """Returns a float_list from a float list."""
#     return tf.train.Feature(float_list=tf.train.FloatList(value=value))

# def _float_feature(value):
#     """Returns a float_list from a float / double."""
#     return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

# def _int64_feature(value):
#     """Returns an int64_list from a bool / enum / int / uint."""
#     return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# def window_example(sig, sbp, dbp, mrn):
#     feature = {
#                'sig': _float_array_feature(sig),
#                'sbp': _float_feature(sbp),
#                'dbp': _float_feature(dbp),
#                'mrn': _int64_feature(mrn),
#               }
#     example = tf.train.Example(features=tf.train.Features(feature=feature))
#     return example

# def write_dataset(file_name, examples: List[tf.train.Example]):
#     with tf.io.TFRecordWriter(file_name) as w:
#         for tf_example in examples:
#             w.write(tf_example.SerializeToString())
#     return

# def _parse_window_function(example_proto):
#     window_feature_description = {
#         'sig': tf.io.FixedLenFeature([625], tf.float32),
#         'sbp': tf.io.FixedLenFeature([], tf.float32),
#         'dbp': tf.io.FixedLenFeature([], tf.float32),
#         'mrn': tf.io.FixedLenFeature([], tf.int64)
#     }
#     return tf.io.parse_single_example(example_proto, window_feature_description)

# def read_dataset(path, n_cores=1):
#     # path is location of TFRecords files.
#     filenames = [file for file in glob.glob(f'{path}*.tfrecords')]
#     print(filenames)
#     dataset = tf.data.TFRecordDataset(filenames=filenames,
#                                       compression_type=None,
#                                       buffer_size=None,
#                                       num_parallel_reads=n_cores)
#     parsed_dataset = dataset.map(_parse_window_function)
#     return parsed_dataset

# def append_sample_count(data_profile_csv, mrn, n_samples):
#     with open(data_profile_csv, 'r') as f:
#         idx = int(f.readlines()[-1].split(',')[0]) + 1
#     with open(data_profile_csv, 'a') as f:
#         w = writer(f)
#         row = [idx, mrn, n_samples]
#         w.writerow(row)    
#     return
