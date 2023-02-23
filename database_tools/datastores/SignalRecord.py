import wfdb
from dataclasses import dataclass, InitVar, field
from database_tools.errors import RecordFormatError

@dataclass
class SignalRecord:

    rcd: InitVar[wfdb.Record]
    fmt: list = field(init=False)
    fs: list = field()
