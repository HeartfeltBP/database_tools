import numpy as np
import tensorflow as tf
from database_tools.tools import RecordsHandler

class TestingPipeline():
    def __init__(self):
        return

    def run():
        return

    def _read_test_data():
        return

    def _test():
        return

def read_test_data(data_dir, n_cores):
    handler = RecordsHandler(data_dir=data_dir)

    dataset = handler.read_records(n_cores=n_cores, AUTOTUNE=tf.data.AUTOTUNE)
    test = dataset['test']
    
    ppg = []
    abp = []
    for p, a in test.as_numpy_iterator():
        ppg.append(p)
        abp.append(a)
    ppg = np.array(ppg).reshape(-1, 256)
    abp = np.array(abp).reshape(-1, 256)
    return dict(ppg=ppg, abp=abp)
