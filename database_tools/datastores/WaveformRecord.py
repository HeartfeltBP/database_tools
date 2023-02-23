import wfdb
from dataclasses import dataclass, InitVar, field
from database_tools.datastores import SignalStore


@dataclass
class WaveformRecord:
    """Interface for wfdb waveforms records.
       Makes it easier to peak at data and build signal
       processing pipelines.

    Attributes
    ----------
    name (str): Record name.
    fs (int): Sampling rate of wfdb.Record.
    """

    rcd: InitVar[wfdb.Record]
    name: str = field(init=False)
    fs: int = field(init=False)

    def __post_init__(self, rcd: wfdb.Record):
        setattr(self, 'name', rcd.record_name)
        setattr(self, 'fs', rcd.fs)
        for i, (sig, fmt) in enumerate(zip(rcd.sig_name, rcd.fmt)):
            setattr(self, sig, SignalStore(rcd.p_signal[:, i], fmt)) 
