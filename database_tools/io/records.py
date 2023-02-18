import io
import wfdb
import logging
import requests
import numpy as np
import pandas as pd
from typing import Union

logging.basicConfig(
     filename='io.log',
     format= '[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
     datefmt='%H:%M:%S'
)
logger = logging.getLogger('io')
logger.setLevel(logging.INFO)

MIMIC_DIR = 'mimic3wdb/1.0/'

def generate_record_paths(name: str = None) -> str:
    if name is None:
        rec_dir = MIMIC_DIR + 'RECORDS'
    elif name == 'adults':
        rec_dir = MIMIC_DIR + 'RECORDS-adults'
    elif name == 'neonates':
        rec_dir = MIMIC_DIR + 'RECORDS-neonates'
    r = requests.get('https://physionet.org/files/' + rec_dir, stream=True)
    records = set(pd.read_csv(io.BytesIO(r.content), names=['records'])['records'])
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
        lay = wfdb.rdheader(pn_dir=pn_dir, record_name=hea_name)
        logger.info(f'Successfully got {record_type} record {hea_name}')
        return lay
    except Exception as e:
        logger.info(f'Failed to get {record_type} record {path} due to {e}')

def get_data_record(path: str) -> wfdb.Record:
    """Get data record from MIMIC-III Waveforms database.

    Args:
        path (str): Path to data file.

    Returns:
        rec (wfdb.io.record.Record): WFDB record object.
    """
    pn_dir = MIMIC_DIR + '/'.join(path.split('/')[:-1])
    rcd_name = path.split('/')[-1]
    try:
        rcd = wfdb.rdrecord(pn_dir=pn_dir, record_name=rcd_name)
        logger.info(f'Successfully got data record {rcd_name}')
        return rcd
    except Exception as e:
        logger.info(f'Failed to get data record {path} due to {e}')

def get_signal(rec: wfdb.Record, sig: str, errors: str = 'ignore') -> np.ndarray:
    """Extract signal data from a record.

    Args:
        rec (wfdb.Record): MIMIC-III record.
        sig (str): Signal name. One of ['PLETH', 'ABP'].
        errors (str, optional): One of ['raise', 'ignore'].

    Returns:
        data (np.ndarray): 1D signal data.
    """
    try:
        s = rec.sig_name  # list of signals in record
        data = rec.p_signal[:, s.index(sig)].astype(np.float64)
        return data
    except Exception as e:
        if errors == 'ignore':
            logger.info(f'Failed to extraction signal due to {e}')
        elif errors == 'raise':
            raise ValueError(f'Signal name \'{sig}\' is not in the provided record')
        else:
            raise ValueError('errors must be one of [\'ignore\', \'raise\']')
