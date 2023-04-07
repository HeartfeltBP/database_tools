import os
import logging
import numpy as np
import plotly.graph_objects as go
from dataclasses import dataclass, InitVar, field

log_dir = '/'.join(os.getcwd().split('/')[0:5]) + '/logs/'
logging.basicConfig(
     filename=f'{log_dir}datastores.log',
     format= '[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
     datefmt='%H:%M:%S'
)
logger = logging.getLogger('datastores')
logger.setLevel(logging.INFO)

SIG_MAP = {
    'PLETH': 'ppg',
    'ABP': 'abp',
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
       Currently support and ppg, abp waveforms.
    """

    signals: InitVar[dict]
    ppg: SignalStore = field(init=False)
    abp: SignalStore = field(init=False)
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
