import matplotlib.pyplot as plt
import numpy as np

from brainmri.image import Image


class MRISlice(Image):

    slice_nr: int
    axis: str

    def __init__(self, slice_nr: int, axis: str, data: np.ndarray):
        self.slice_nr = slice_nr
        self.data = data
        self.axis = axis

    def plot(self, return_fig=False):
        return plot_slice(self, return_fig=return_fig)

    def volume_in_ml(self, voxel_size, intensity=None) -> float:
        return volume_in_ml_slice(self, voxel_size, intensity=intensity)


def volume_in_ml_slice(slice: MRISlice, voxel_size, intensity=None) -> float:
    if intensity == 0:
        raise ValueError("Intensity must be greater than 0")

    if intensity is None:
        mask = slice.data > 0
    else:
        mask = slice.data == intensity
    slice_voxels = np.count_nonzero(mask)
    slice_volume = slice_voxels * voxel_size
    return slice_volume / 1000


def plot_slice(slice: MRISlice, return_fig=False):

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(slice.data, cmap="gray")
    if return_fig:
        return fig
