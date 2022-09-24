import numpy as np
import pandas as pd
from plotting import histogram3d


class DataEvaluator():
    def __init__(self,
                 sim,
                 snr,
                 hr,
                 n_excluded):
        self._sim = sim
        self._snr = snr
        self._hr = hr
        self._n_excluded = n_excluded

    def evaluate_sim(self, bins=50):
        pleth, abp = self._sim[:, 0], self._sim[:, 1]
        fig = histogram3d(pleth, abp, range_=[[0, 1], [0, 1]], title='Similarity', bins=bins)
        fig.update_scenes(yaxis_autorange='reversed')
        return fig

    def evaluate_snr(self, bins=50):
        pleth_snr = pd.Series(self._snr[:, 0])
        abp_snr = pd.Series(self._snr[:, 1])
        pleth  = pleth_snr.plot.hist(nbins=bins)
        abp = abp_snr.plot.hist(nbins=bins)
        return (pleth, abp)

    def evaluate_hr(self, bins=50):
        pleth, abp = self._hr[:, 0] * 60, self._hr[:, 1] * 60
        min_ = np.min((pleth, abp)) - 10
        max_ = np.max((pleth, abp)) + 10
        fig = histogram3d(pleth, abp, range_=[[min_, max_], [min_, max_]], title='Heart Rate', bins=bins)
        fig.update_scenes(xaxis_range=[min_, max_],
                          yaxis_range=[min_, max_],
                          yaxis_autorange='reversed')
        return fig

    def percent_dropped(self):
        return self._n_excluded / (len(self._sim) + self._n_excluded)
