import wfdb
from typing import List

def layout_has_signals(hea: wfdb.Record, signals: List[str]) -> bool:
    """Check if a patient layout contains specified signals.

    Args:
        hea (wfdb.Record): Layout header file as wfdb.Record object.
        signals (List[str]): List of signals.

    Returns:
        bool: True if the layout contains ALL signals provided.
    """
    if hea is None:
        return False
    elif set(signals).difference(set(hea.sig_name)) == set():
        return True
    else:
        return False
