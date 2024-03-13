from typing import List

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objs as go

from brainmri.constants import INTENSITY_TO_NAME_MAPPING
from brainmri.core.mri_image import MRIImage
from brainmri.core.mri_slice import MRISlice


def volume_histogram(volumes, intensities):

    fig = px.bar(
        x=[INTENSITY_TO_NAME_MAPPING[str(int(i))] for i in intensities],
        y=volumes,
        title="Volume per Intensity",
    )
    fig.update_layout(
        xaxis_title="Intensity",
        yaxis_title="Volume (ml)",
        showlegend=False,
    )
    return fig


def plot_3d_brain(mri_image: MRIImage, to_slide_nr: int = None):

    # Assuming slices is a list of 2D numpy arrays
    slices = mri_image.slices_z[::2]

    stack = []

    for i, slice in enumerate(slices[:to_slide_nr]):
        image = slice.data
        x, y = np.where(image > 0)  # retain this to limit to non-zero elements
        z = [i * 2] * len(x)
        intensity = image[x, y]  # get the intensity values
        for xi, yi, zi, intensity_i in zip(x, y, z, intensity):
            stack.append([xi, yi, zi, intensity_i])

    stack = np.array(stack)

    # Identify unique intensities and corresponding indices
    unique_intensities, inverse = np.unique(stack[:, 3], return_inverse=True)

    # Use colormap to derive colors
    cmap = plt.cm.get_cmap(
        "jet", len(unique_intensities)
    )  # 'jet' colormap with number of unique_intensities' colors
    colors = cmap(inverse, bytes=True)

    fig = go.Figure(
        data=go.Scatter3d(
            x=stack[:, 0],
            y=stack[:, 1],
            z=stack[:, 2],
            mode="markers",
            marker=dict(
                size=5,
                color=[
                    "rgba" + str(tuple(val)) for val in colors
                ],  # set color to an array/list of desired values
                opacity=0.01,
            ),
        )
    )

    fig.update_scenes(
        xaxis=dict(range=[0, 100]),  # replace with desired x-axis limits
        yaxis=dict(range=[0, 100]),  # replace with desired y-axis limits
        zaxis=dict(range=[0, 100]),  # replace with desired z-axis limits
    )

    return fig


def plot_animator(slices: List[MRISlice]):
    slice_images = [slice.data for slice in slices]
    fig = px.imshow(np.array(slice_images), animation_frame=0, binary_string=True)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(autosize=False, width=500, height=500, coloraxis_showscale=False)
    # Drop animation buttons
    fig["layout"].pop("updatemenus")
    return fig
