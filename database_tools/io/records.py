import wfdb
import logging
import numpy as np

logging.basicConfig(
     filename='io.log',
     format= '[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
     datefmt='%H:%M:%S'
)
logger = logging.getLogger('io')
logger.setLevel(logging.INFO)

def get_layout_record(path: str) -> wfdb.Record:
    """Get layout record from MIMIC-III Waveforms database.

    Args:
        path (str): Path to data file.

    Returns:
        rec (wfdb.io.record.Record): WFDB record object.
    """
    dir = 'mimic3wdb/1.0/' + path
    hea_name = path.split('/')[1] + '_layout'
    try:
        hea = wfdb.rdheader(pn_dir=dir, record_name=hea_name)
        logger.info(f'Successfully got layout record {hea_name}')
        return hea
    except Exception as e:
        logger.info(f'Failed to get layout record {path} due to {e}')

def get_data_record(path: str) -> wfdb.Record:
    """Get data record from MIMIC-III Waveforms database.

    Args:
        path (str): Path to data file.

    Returns:
        rec (wfdb.io.record.Record): WFDB record object.
    """
    dir = 'mimic3wdb/1.0/' + '/'.join(path.split('/')[:-1])
    rcd_name = path.split('/')[-1]
    try:
        rcd = wfdb.rdrecord(pn_dir=dir, record_name=rcd_name)
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
            raise ValueError('errors must be one of \'ignore\', \'raise\'')
