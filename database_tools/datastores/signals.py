import numpy as np
from typing import List, Tuple
import plotly.graph_objects as go
from dataclasses import dataclass, InitVar, field
from database_tools.filtering.utils import ConfigMapper, repair_peaks_troughs_idx
from database_tools.filtering.functions import get_snr, flat_lines, beat_similarity, find_peaks, detect_notches, get_similarity, bandpass

SIG_MAP = {
    'PLETH': 'ppg',
    'ABP': 'abp',
    'II': 'ecg_ii',
}


@dataclass
class SignalStore:
    """Object to store wfdb.Record signal data and format."""

    data: np.ndarray
    fmt: str

    def replace_nan(self, value: float = 1.0):
        """Replace NaN values in the signal.

        Args:
            value (float): Replacement value. Defaults to 1.0.

        Raises:
            TypeError: Param value must be type float.
        """
        if not isinstance(value, float):
            raise TypeError('replace_nan: value must be type float')
        self.data[np.isnan(self.data)] = value

    def plot(self, mode: str = 'line', show: bool = True, **kwargs):
        """Plot signal data.

        Args:
            mode (str, optional): One of ['line', 'hist']. Defaults to 'line'.
            show (bool, optional): Show the plot. If False plot is returned. Defaults to True.
            **kwargs: Plotly graph parameters.

        Returns:
            fig (plotly.Figure): Plotly figure.
        """
        fig = go.Figure()
        if mode == 'line':
            fig.add_scatter(y=self.data, **kwargs)
        elif mode == 'hist':
            fig.add_histogram(x=self.data, **kwargs)
        if show:
            fig.show()
        else:
            return fig


@dataclass
class SignalGroup:
    """Object to store a group of SignalStore objects.
       Currently support ppg, abp, and ecg_ii waveforms.
    """

    signals: InitVar[dict]
    ppg: SignalStore = field(init=False)
    abp: SignalStore = field(init=False)
    ecg_ii: SignalStore = field(init=False)
    _names: list = field(init=False)

    def __post_init__(self, signals):
        names = []
        for sig, store in signals.items():
            setattr(self, SIG_MAP[sig], store)
            names.append(SIG_MAP[sig])
        setattr(self, '_names', names)

    def info(self):
        """Print signal names, formats and lengths."""
        out = ''
        for sig in self._names:
            store = getattr(self, sig)
            out += f'{sig: <7}: '
            out += f'Format {store.fmt}, Length of {store.data.shape[0]} samples\n'
        print(out)


@dataclass
class Window:

    sig: np.ndarray
    cm: ConfigMapper
    checks: List[str]

    @property
    def _snr_check(self) -> bool:
        self.snr, self.f0 = get_snr(self.sig, low=self.cm.freq_band[0], high=self.cm.freq_band[1], df=0.2, fs=self.cm.fs)
        return self.snr > self.cm.snr

    @property
    def _hr_check(self) -> bool:
        return (self.f0 > self.cm.hr_freq_band[0]) & (self.f0 < self.cm.hr_freq_band[1])

    @property
    def _flat_check(self) -> bool:
        return not flat_lines(self.sig, n=self.cm.flat_line_length)

    @property
    def _beat_check(self) -> bool:
        self.beat_sim = beat_similarity(
            self.sig,
            troughs=self.troughs,
            fs=self.cm.fs,
        )
        return self.beat_sim > self.cm.beat_sim

    @property
    def _notch_check(self) -> bool:
        notches = detect_notches(
            self.sig,
            peaks=self.peaks,
            troughs=self.troughs,
        )
        return len(notches) > self.cm.min_notches

    @property
    def _bp_check(self) -> bool:
        self.dbp, self.sbp = np.min(self.sig), np.max(self.sig)
        dbp_check = (self.dbp > self.cm.dbp_bounds[0]) & (self.dbp < self.cm.dbp_bounds[1])
        sbp_check = (self.sbp > self.cm.sbp_bounds[0]) & (self.sbp < self.cm.sbp_bounds[1])
        return dbp_check & sbp_check

    @property
    def valid(self) -> bool:
        v = [object.__getattribute__(self, '_' + c + '_check') for c in self.checks]
        return np.array(v).all()

    def get_peaks(self, pad_width=40) -> None:
        x_pad = np.pad(self.sig, pad_width=pad_width, constant_values=np.mean(self.sig))
        peaks, troughs = find_peaks(x_pad).values()
        peaks, troughs = repair_peaks_troughs_idx(peaks, troughs)
        self.peaks = peaks - pad_width - 1
        self.troughs = troughs - pad_width - 1

def congruency_check(ppg: Window, abp: Window, cm: ConfigMapper) -> Tuple[float, float, bool]:
    """Performs checks between ppg and abp windows.

    Args:
        ppg (Window): Object with ppg data.
        abp (Window): Object with abp data.
        cm (ConfigMapper): Config mapping dataclass.

    Returns:
        bool: True if valid, False if not.
    """
    time_sim = get_similarity(ppg.sig, abp.sig)
    ppg_f = np.abs(np.fft.fft(ppg.sig))
    abp_f = np.abs(np.fft.fft(bandpass(abp.sig, low=cm.freq_band[0], high=cm.freq_band[1], fs=cm.fs)))
    spec_sim = get_similarity(ppg_f, abp_f)
    sim_check = (time_sim > cm.sim) & (spec_sim > cm.sim)
    hr_delta_check = np.abs(ppg.f0 - abp.f0) < cm.hr_delta
    congruency_check = sim_check & hr_delta_check
    return (time_sim, spec_sim, congruency_check)
