from threading import stack_size
import numpy as np
import pandas as pd
from database_tools.plotting import histogram3d

pd.options.plotting.backend = 'plotly'

class DataEvaluator():
    def __init__(self, stats):
        self._stats = stats
        self._stats = self._stats.fillna(0)
        self._sim = np.concatenate(
            (
                np.array(self._stats['time_similarity']).reshape(-1, 1),
                np.array(self._stats['spectral_similarity']).reshape(-1, 1),
            ),
            axis=1
        )
        self._snr = np.concatenate(
            (
                np.array(self._stats['ppg_snr']).reshape(-1, 1),
                np.array(self._stats['abp_snr']).reshape(-1, 1),
            ),
            axis=1,
        )
        self._hr = np.concatenate(
            (
                np.array(self._stats['ppg_hr']).reshape(-1, 1),
                np.array(self._stats['abp_hr']).reshape(-1, 1),
            ),
            axis=1,
        )

    def run(self):
        figs = {}
        figs['sim'] = self.evaluate_sim()
        fig_p, fig_a = self.evaluate_snr()
        figs['snr_pleth'] = fig_p
        figs['snr_abp'] = fig_a
        figs['hr'] = self.evaluate_hr()
        return figs

    def evaluate_sim(self, bins=50):
        pleth, abp = self._sim[:, 0], self._sim[:, 1]
        fig = histogram3d(pleth, abp, range_=[[0, 1], [0, 1]], title='Similarity', bins=bins)
        fig.update_scenes(yaxis_autorange='reversed')
        return fig

    def evaluate_snr(self, bins=50):
        pleth_snr = pd.Series(self._snr[:, 0])
        abp_snr = pd.Series(self._snr[:, 1])
        fig_p  = pleth_snr.plot.hist(nbins=bins)
        fig_a = abp_snr.plot.hist(nbins=bins)
        return (fig_p, fig_a)

    def evaluate_hr(self, bins=25):
        pleth, abp = self._hr[:, 0], self._hr[:, 1]
        min_ = np.min((pleth, abp)) - 10
        max_ = np.max((pleth, abp)) + 10
        fig = histogram3d(pleth, abp, range_=[[min_, max_], [min_, max_]], title='Heart Rate', bins=bins)
        fig.update_scenes(yaxis_range=[max_, min_],
                          xaxis_range=[min_, max_],
                         )
        return fig
