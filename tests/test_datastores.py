import plotly
import numpy as np
from database_tools.io.records import get_data_record
from database_tools.datastores import SignalStore, WaveformRecord, NumericsRecord

def test_SignalStore():
    rcd = get_data_record('30/3000063/3000063_0016', 'waveforms')
    sig = rcd.p_signal[0]
    sig[0] = np.nan
    store = SignalStore(sig, rcd.fmt[0])
    store.replace_nan(value=-999.0)
    fig = store.plot(mode='line', show=False)
    assert isinstance(store(), np.ndarray)
    assert store.shape == sig.shape
    assert store()[0] == -999.0
    assert isinstance(fig, plotly.graph_objs.Figure)

def test_WaveformRecord():
    tmp = get_data_record('30/3000063/3000063_0016', 'waveforms')
    rcd = WaveformRecord(tmp)
    assert rcd.name == '3000063_0016'

def test_NumericsRecord():
    pass

