import numpy as np
import plotly.graph_objects as go
from dataclasses import dataclass


@dataclass
class SignalStore:
    """Object to store wfdb.Record signal data and format."""

    data: np.ndarray
    fmt: str

    def __call__(self):
        """
        Returns:
            np.ndarray: Signal data.
        """
        return self.data

    @property
    def shape(self):
        """
        Returns:
            Tuple: np.ndarray.shape
        """
        return self.data.shape

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
