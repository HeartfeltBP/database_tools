import numpy as np
import pandas as pd
from database_tools.plotting import histogram3d

pd.options.plotting.backend = 'plotly'

# TODO Add plot titles
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
        figs['snr_ppg'] = fig_p
        figs['snr_abp'] = fig_a
        figs['hr'] = self.evaluate_hr()
        return figs

    def evaluate_sim(self, bins=50):
        ppg, abp = self._sim[:, 0], self._sim[:, 1]
        fig = histogram3d(ppg, abp, range_=[[0, 1], [0, 1]], title='Similarity', bins=bins)
        fig.update_scenes(yaxis_autorange='reversed')
        return fig

    def evaluate_snr(self, bins=100):
        ppg_snr = pd.Series(self._snr[:, 0])
        abp_snr = pd.Series(self._snr[:, 1])
        abp_snr[abp_snr < -30] = -30
        fig_p  = ppg_snr.plot.hist(nbins=bins)
        fig_a = abp_snr.plot.hist(nbins=bins)
        return (fig_p, fig_a)

    def evaluate_hr(self, bins=25):
        ppg, abp = self._hr[:, 0], self._hr[:, 1]

        # Exclude extreme values
        ppg[ppg > 250] = 0
        abp[abp > 250] = 0

        min_ = np.min([np.min(ppg), np.min(abp)]) - 10
        max_ = np.max([np.max(ppg), np.max(abp)]) + 10
        fig = histogram3d(ppg, abp, range_=[[min_, max_], [min_, max_]], title='Heart Rate', bins=bins)
        fig.update_scenes(yaxis_range=[max_, min_],
                          xaxis_range=[min_, max_],
                         )
        return fig
