import numpy as np
import plotly.graph_objects as go

def bland_altman_plot(data1, data2):
    data1     = np.asarray(data1)
    data2     = np.asarray(data2)
    mean      = np.mean([data1, data2], axis=0)
    diff      = data1 - data2                   # Difference between data1 and data2
    md        = np.mean(diff)                   # Mean of the difference
    sd        = np.std(diff, axis=0)            # Standard deviation of the difference

    fig = go.Figure()
    fig.add_scatter(x=mean, y=diff, mode='markers')
    fig.add_hline(y=md)
    for i in [5, 10, 15]:
        fig.add_hline(y=md + i, line_dash='dot')
        fig.add_hline(y=md - i, line_dash='dot')
    fig.update_layout(
        yaxis=dict(
            range=[-55, 55],
        )
    )
    return fig
