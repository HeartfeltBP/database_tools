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
        figs['ppg_snr'] = fig_p
        figs['abp_snr'] = fig_a
        figs['hr'] = self.evaluate_hr()
        return figs

    def evaluate_sim(self, bins=30):
        ppg, abp = self._sim[:, 0], self._sim[:, 1]
        fig = histogram3d(ppg, abp, range_=[[0, 1], [0, 1]], bins=bins)
        fig.update_layout(
                title={
                    'text': 'PPG, ABP Similarity Histogram',
                    'font': {'size': 35},
                },
                scene=dict(
                    xaxis={'title': 'Time Similarity (rₜ)', 'titlefont':{'size': 20}},
                    yaxis={'title': 'Spectral Similarity (r𝓌)', 'titlefont':{'size': 20}},
                    zaxis={'title': 'Number of Samples', 'titlefont':{'size': 20}},
                ),
                font={
                    'family': 'Courier New, monospace',
                    'color' : '#FFFFFF',
                    'size'  : 12,
                },
                template='plotly_dark',
        )
        fig.update_scenes(yaxis_autorange='reversed')
        return fig

    def evaluate_snr(self, bins=100):
        ppg_snr = pd.Series(self._snr[:, 0])
        abp_snr = pd.Series(self._snr[:, 1])
        ppg_snr[ppg_snr < -10] = -10
        abp_snr[abp_snr < -10] = -10  # modify extreme values for plotting
        fig_p  = ppg_snr.plot.hist(nbins=bins)
        fig_p.update_layout(
            title='PPG Signal-to-Noise Ratio Histogram',
            xaxis={'title': 'PPG SNR (dB)'},
            yaxis={'title': 'Number of Samples'},
            showlegend=False,
            template='plotly_dark',
        ) 
        fig_a = abp_snr.plot.hist(nbins=bins)
        fig_a.update_layout(
            title='ABP Signal-to-Noise Ratio Histogram',
            xaxis={'title': 'ABP SNR (dB)'},
            yaxis={'title': 'Number of Samples'},
            showlegend=False,
            template='plotly_dark',
        ) 
        return (fig_p, fig_a)

    def evaluate_hr(self, bins=25):
        ppg, abp = self._hr[:, 0], self._hr[:, 1]

        # Exclude extreme values
        ppg[ppg > 250] = 0
        abp[abp > 250] = 0

        min_ = np.min([np.min(ppg), np.min(abp)]) - 10
        max_ = np.max([np.max(ppg), np.max(abp)]) + 10
        fig = histogram3d(ppg, abp, range_=[[min_, max_], [min_, max_]], bins=bins)
        fig.update_layout(
                title={
                    'text': 'PPG, ABP HR Histogram',
                    'font': {'size': 35},
                },
                scene=dict(
                    xaxis={'title': 'PPG Heart Rate', 'titlefont':{'size': 20}},
                    yaxis={'title': 'ABP Heart Rate', 'titlefont':{'size': 20}},
                    zaxis={'title': 'Number of Samples', 'titlefont':{'size': 20}},
                ),
                font={
                    'family': 'Courier New, monospace',
                    'color' : '#FFFFFF',
                    'size'  : 12,
                },
                template='plotly_dark',
        )
        fig.update_scenes(
            yaxis_range=[max_, min_],
            xaxis_range=[min_, max_],
        )
        return fig
