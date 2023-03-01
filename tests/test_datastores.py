import plotly
import numpy as np
from database_tools.io.wfdb import get_data_record
from database_tools.datastores.signals import SignalStore, SignalGroup
from database_tools.datastores.records import WaveformRecord, NumericsRecord

def test_SignalStore():
    rcd = get_data_record('30/3000063/3000063_0016', 'waveforms')
    sig = rcd.p_signal[0]
    sig[0] = np.nan
    store = SignalStore(sig, rcd.fmt[0])
    store.replace_nan(value=-999.0)
    fig = store.plot(mode='line', show=False)
    assert isinstance(store.data, np.ndarray)
    assert store.data.shape == sig.shape
    assert store.data[0] == -999.0
    assert isinstance(fig, plotly.graph_objs.Figure)

def test_SignalGroup():
    rcd = get_data_record('30/3000063/3000063_0016', 'waveforms')
    signals = {sig: SignalStore(rcd.p_signal[:, i], rcd.fmt[i]) for i, sig in enumerate(rcd.sig_name)}
    store = SignalGroup(signals)
    assert isinstance(store.ppg, SignalStore)
    assert isinstance(store.abp, SignalStore)
    assert isinstance(store.ecg_ii, SignalStore)

def test_WaveformRecord():
    tmp = get_data_record('30/3000063/3000063_0016', 'waveforms')
    rcd = WaveformRecord(tmp)
    assert rcd.name == '3000063_0016'

def test_NumericsRecord():
    pass

