import os
import io
import wfdb
import random
import logging
import requests
import numpy as np
import pandas as pd
from typing import Union, List
from alive_progress import alive_bar

try:
    logging.basicConfig(
        filename=f'{__name__}.log',
        format= '[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    LOG = True
except OSError:
    LOG = False

MIMIC_DIR = 'mimic3wdb/1.0/'

def generate_record_paths(name: str = None, shuffle: bool = True) -> str:
    if name is None:
        rec_dir = MIMIC_DIR + 'RECORDS'
    elif name == 'adults':
        rec_dir = MIMIC_DIR + 'RECORDS-adults'
    elif name == 'neonates':
        rec_dir = MIMIC_DIR + 'RECORDS-neonates'
    r = requests.get('https://physionet.org/files/' + rec_dir, stream=True)
    records = list(pd.read_csv(io.BytesIO(r.content), names=['records'])['records'])
    if shuffle:
        random.shuffle(records)
    if LOG: logger.info(f'Successfuly got records file from {rec_dir}')
    for path in records:
        yield path[:-1]  # remove trailing /

def get_header_record(path: str, record_type: str) -> Union[wfdb.Record, wfdb.MultiRecord]:
    """Get data header or layout header record from MIMIC-III Waveforms database.

    Args:
        path (str): Path to data file.
        record_type (str): One of ['layout', 'data'].

    Returns:
        rec (wfdb.io.record.Record): WFDB record object.
    """
    pn_dir = MIMIC_DIR + path
    if record_type == 'layout':
        hea_name = path.split('/')[-1] + '_layout'
    elif record_type == 'data':
        pn_dir = pn_dir.rsplit('/', 1)[0] if '_' in pn_dir else pn_dir  # remove file name from path if there
        hea_name = path.split('/')[-1]
    else:
        raise ValueError('record_type must be one of [\'layout\', \'data\']')

    try:
        hea = wfdb.rdheader(pn_dir=pn_dir, record_name=hea_name)
        if LOG: logger.info(f'Successfully got {record_type} record {hea_name}')
        return hea
    except Exception as e:
        if LOG: logger.info(f'Failed to get {record_type} record {path} due to {e}')

def header_has_signals(hea: wfdb.Record, signals: List[str]) -> bool:
    """Check if a header record contains specified signals.

    Args:
        hea (wfdb.Record): Header file as wfdb.Record object.
        signals (List[str]): List of signals.

    Returns:
        bool: True if the record contains ALL signals provided.
    """
    try:
        if set(signals).difference(set(hea.sig_name)) == set():
            if LOG: logger.info(f'Record {hea.record_name}.hea contains {signals}')
            return True
        else:
            if LOG: logger.info(f'Record {hea.record_name}.hea does not contain {signals}')
            return False
    except TypeError:
        if LOG: logger.info(f'Error getting {signals} from {hea.record_name}.hea')
        return False

def get_data_record(path: str, record_type: str = 'waveforms') -> wfdb.Record:
    """Get data record from MIMIC-III Waveforms database.

    Args:
        path (str): Path to data file.
        record_type (str): One of ['waveforms', 'numerics']. Defaults to 'numerics'.

    Returns:
        rec (wfdb.io.record.Record): WFDB record object.
    """
    if record_type == 'waveforms':
        pn_dir = MIMIC_DIR + '/'.join(path.split('/')[:-1])
        rcd_name = path.split('/')[-1]
    elif record_type == 'numerics':
        pn_dir = MIMIC_DIR + path
        rcd_name = path.split('/')[-1] + 'n'
    else:
        raise ValueError('record_type must be one of [\'waveforms\', \'numerics\']')
    try:
        rcd = wfdb.rdrecord(pn_dir=pn_dir, record_name=rcd_name)
        if LOG: logger.info(f'Successfully got {record_type} record {rcd_name}')
        return rcd
    except Exception as e:
        if LOG: logger.info(f'Failed to get {record_type} record {path} due to {e}')

def get_signal(rec: wfdb.Record, sig: str, errors: str = 'ignore') -> np.ndarray:
    """Extract signal data from a record.

    Args:
        rec (wfdb.Record): MIMIC-III record.
        sig (str): Signal name. One of ['PLETH', 'ABP'].
        errors (str, optional): One of ['raise', 'ignore'].

    Returns:
        data (np.ndarray): 1D signal data or NoneType if errors is 'ignore'.
    """
    try:
        s = rec.sig_name  # list of signals in record
        data = rec.p_signal[:, s.index(sig)].astype(np.float64)
        if LOG: logger.info(f'Successfully extracted {sig} from {rec.record_name}')
        return data
    except Exception as e:
        if errors == 'ignore':
            if LOG: logger.info(f'Failed to extract {sig} from {rec.record_name} due to {e}')
        elif errors == 'raise':
            raise ValueError(f'Signal name {sig} is not in the provided record')
        else:
            raise ValueError('errors must be one of [\'ignore\', \'raise\']')

def locate_valid_records(signals: List[str], min_length: int, n_segments: int, shuffle: bool = True) -> List[str]:
    """Locate valid data records. Exclusion is performed based on a list of signals
    the records must contain and a minimum length of the signals.

    Args:
        signals (List[str]): One or more of ['PLETH', 'ABP', ...]  TODO: add other signals
        min_length (int): Minimum length of data records to be considered valid.
        n_segments (int): Maximum number of records to find.
        shuffle (bool): If True records list is shuffled.

    Returns:
        valid_records (List[str]): List of valid segments.
    """
    valid_records = []
    with alive_bar(total=n_segments, bar='brackets', force_tty=True) as bar:
        for path in generate_record_paths(name='adults', shuffle=shuffle):

            # get patient layout header
            layout = get_header_record(path=path, record_type='layout')
            if layout is None: continue  # fx returns None if file DNE

            # check if header has provided signals
            if not header_has_signals(layout, signals): continue

            # get patient master header
            master = get_header_record(path=path, record_type='data')
            if master is None: continue

            # zip segment names and lengths
            for seg_name, n_samples in zip(master.seg_name, master.seg_len):
            
                # check segment length
                if (n_samples > min_length) & (seg_name != '~'):  # '~' indicates data is missing
                    seg_path = path + '/' + seg_name

                    # Get segment header
                    hea = get_header_record(path=seg_path, record_type='data')
                    if hea is None: continue

                    # Check if segment has provided signals and append
                    if header_has_signals(hea, signals):
                        valid_records.append(seg_path)
                        if n_segments is not None:
                            if len(valid_records) > n_segments:
                                return valid_records
                        bar()  # iterate loading bar
