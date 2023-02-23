import wfdb
from dataclasses import dataclass, InitVar, field
from database_tools.datastores import SignalRecord
from database_tools.errors import RecordFormatError

@dataclass
class WaveformRecord:

    rcd: InitVar[wfdb.Record]
    name: str = field(init=False)
    signals: SignalRecord = field(init=False)

    def __post_init__(self, rcd: wfdb.Record):
        self.__setattr__(object, 'name', rcd.record_name)
        signals = {sig: rcd.p_signal[:, i] for i, sig in enumerate(rcd.p_signal)}
        self.__setattr__(object, 'signals', signals)
        self.__setattr__(object, 'fs', rcd.fs)