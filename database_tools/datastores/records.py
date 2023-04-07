import wfdb
from dataclasses import dataclass, InitVar, field
from database_tools.datastores.signals import SignalStore, SignalGroup


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
        signals = {sig: SignalStore(rcd.p_signal[:, i], rcd.fmt[i]) for i, sig in enumerate(rcd.sig_name) if sig in ['PLETH', 'ABP']}
        group = SignalGroup(signals)
        setattr(self, 'sigs', group)


@dataclass
class NumericsRecord:
    def __post_init__(self):
        #TODO:
        pass
