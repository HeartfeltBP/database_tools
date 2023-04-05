import numpy as np
import pandas as pd
import plotly.graph_objects as go

pd.options.plotting.backend = 'plotly'

class DataEvaluator():
    def __init__(self, stats):
        self._stats = stats
        self._stats = self._stats.fillna(0)
        self._sim = np.concatenate(
            (
                np.array(self._stats['time_sim']).reshape(-1, 1),
                np.array(self._stats['spec_sim']).reshape(-1, 1),
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
        self._bp = np.concatenate(
            (
                np.array(self._stats['sbp']).reshape(-1, 1),
                np.array(self._stats['dbp']).reshape(-1, 1),
            ),
            axis=1,
        )
        self._beat_sim = np.concatenate(
            (
                np.array(self._stats['ppg_beat_sim']).reshape(-1, 1),
                np.array(self._stats['abp_beat_sim']).reshape(-1, 1),
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
        fig_sbp, fig_dbp = self._evaluate_bp()
        figs['sbp'] = fig_sbp
        figs['dbp'] = fig_dbp
        fig_beat_sim_p, fig_beat_sim_a = self.evaluate_beat_sim()
        figs['ppg_beat_sim'] = fig_beat_sim_p
        figs['abp_beat_sim'] = fig_beat_sim_a
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
        ppg_snr[ppg_snr > 20] = 20
        abp_snr[abp_snr > 20] = 20
        abp_snr[abp_snr > 20] = 20 # modify extreme values for plotting

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

    def evaluate_beat_sim(self, bins=30):
        ppg = pd.Series(self._beat_sim[:, 0])
        abp = pd.Series(self._beat_sim[:, 1])
        fig_p  = ppg.plot.hist(nbins=bins)
        fig_a = abp.plot.hist(nbins=bins)
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

    def _evaluate_bp(self, bins=100):
        sbp = pd.Series(self._bp[:, 0])
        dbp = pd.Series(self._bp[:, 1])
        # sbp[sbp < -10] = -10
        # sbp[sbp > 20] = 20
        # dbp[dbp > 20] = 20
        # dbp[dbp > 20] = 20 # modify extreme values for plotting

        fig_sbp  = sbp.plot.hist(nbins=bins)
        fig_sbp.update_layout(
            title='Systolic Blood Pressure Histogram (max by window)',
            xaxis={'title': 'Blood Pressure (mmHg)'},
            yaxis={'title': 'Number of Samples'},
            showlegend=False,
            template='plotly_dark',
        ) 
        fig_dbp = dbp.plot.hist(nbins=bins)
        fig_dbp.update_layout(
            title='Diastolic Blood Pressure Histogram (min by window)',
            xaxis={'title': 'Blood Pressure (mmHg)'},
            yaxis={'title': 'Number of Samples'},
            showlegend=False,
            template='plotly_dark',
        ) 
        return (fig_sbp, fig_dbp)

def _bar_data(position3d, size=(1,1,1)):
    # position3d - 3-list or array of shape (3,) that represents the point of coords (x, y, 0), where a bar is placed
    # size = a 3-tuple whose elements are used to scale a unit cube to get a paralelipipedic bar
    # returns - an array of shape(8,3) representing the 8 vertices of  a bar at position3d
    
    bar = np.array([[0, 0, 0],
                    [1, 0, 0],
                    [1, 1, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [1, 0, 1],
                    [1, 1, 1],
                    [0, 1, 1]], dtype=float) # the vertices of the unit cube

    bar *= np.asarray(size)# scale the cube to get the vertices of a parallelipipedic bar
    bar += np.asarray(position3d) #translate each  bar on the directio OP, with P=position3d
    return bar

def _triangulate_bar_faces(positions, sizes=None):
    # positions - array of shape (N, 3) that contains all positions in the plane z=0, where a histogram bar is placed 
    # sizes -  array of shape (N,3); each row represents the sizes to scale a unit cube to get a bar
    # returns the array of unique vertices, and the lists i, j, k to be used in instantiating the go.Mesh3d class

    if sizes is None:
        sizes = [(1,1,1)]*len(positions)
    else:
        if isinstance(sizes, (list, np.ndarray)) and len(sizes) != len(positions):
            raise ValueError('Your positions and sizes lists/arrays do not have the same length')
            
    all_bars = [_bar_data(pos, size)  for pos, size in zip(positions, sizes) if size[2]!=0]
    p, q, r = np.array(all_bars).shape

    # extract unique vertices from the list of all bar vertices
    vertices, ixr = np.unique(np.array(all_bars).reshape(p*q, r), return_inverse=True, axis=0)
    #for each bar, derive the sublists of indices i, j, k assocated to its chosen  triangulation
    I = []
    J = []
    K = []

    for k in range(len(all_bars)):
        I.extend(np.take(ixr, [8*k, 8*k+2,8*k, 8*k+5,8*k, 8*k+7, 8*k+5, 8*k+2, 8*k+3, 8*k+6, 8*k+7, 8*k+5])) 
        J.extend(np.take(ixr, [8*k+1, 8*k+3, 8*k+4, 8*k+1, 8*k+3, 8*k+4, 8*k+1, 8*k+6, 8*k+7, 8*k+2, 8*k+4, 8*k+6])) 
        K.extend(np.take(ixr, [8*k+2, 8*k, 8*k+5, 8*k, 8*k+7, 8*k, 8*k+2, 8*k+5, 8*k+6, 8*k+3, 8*k+5, 8*k+7]))  

    return  vertices, I, J, K  #triangulation vertices and I, J, K for mesh3d

def _get_plotly_mesh3d(x, y, range_, bins=5, bargap=0.05):
    bins = [bins, bins]
    # x, y- array-like of shape (n,), defining the x, and y-ccordinates of data set for which we plot a 3d hist
    hist, xedges, yedges = np.histogram2d(x, y,
                                          bins=bins,
                                          range=[range_[0],
                                                 range_[1]])
    xsize = xedges[1]-xedges[0]-bargap
    ysize = yedges[1]-yedges[0]-bargap
    xe, ye= np.meshgrid(xedges[:-1], yedges[:-1])
    ze = np.zeros(xe.shape)

    positions = np.dstack((xe, ye, ze))
    m, n, p = positions.shape
    positions = positions.reshape(m*n, p)
    sizes = np.array([(xsize, ysize, h) for h in hist.flatten()])
    vertices, I, J, K  = _triangulate_bar_faces(positions, sizes=sizes)
    X, Y, Z = vertices.T
    return X, Y, Z, I, J, K

def histogram3d(x, y, range_, bins=50):
    X, Y, Z, I, J, K = _get_plotly_mesh3d(x, y, range_, bins=bins, bargap=0)

    lighting = go.mesh3d.Lighting(
        ambient=0.6,
        roughness=0.1,
        specular=1.0,
        fresnel=5.0,
    )
    mesh3d = go.Mesh3d(
        x=X, y=Y, z=Z,
        i=I, j=J, k=K,
        colorscale='Viridis',
        intensity=np.linspace(0, 1, len(X), endpoint=True),
        flatshading=False,
        lighting=lighting,
    )
    layout = go.Layout(width=1000, 
                       height=1000, 
                       scene=dict(
                                  camera_eye_x=-1.0, 
                                  camera_eye_y=1.25,
                                  camera_eye_z=1.25,
                                 ),
                       yaxis={'autorange': False},
                       xaxis={},
                       font={},
                       )
    fig = go.Figure(data=[mesh3d], layout=layout)
    return fig