import numpy as np
import plotly.graph_objects as go


def visualize(target, N=31):
    # Note that the order is reversed because the input is in Fortran type format
    # target should be one-dimensional vector with N^3 elements in Fortran order
    Z, Y, X = np.meshgrid(np.arange(N), np.arange(N), np.arange(N))
    fig = go.Figure(data=go.Volumne(
                    x=X.flatten(),
                    y=Y.flatten(),
                    z=Z.flatten(),
                    value=target,
                    opacityscale=[[0, 1], [1, 0]],  # Set value dependent opacity
                    surface_count=10,  # Too high value will slow the browser
                    colorscale=[[0, 'rgb(255, 255, 255)'], [1, 'rgb(255, 0, 0)']]
                    ))
    fig.show()
