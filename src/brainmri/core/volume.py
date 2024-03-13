from typing import List

import numpy as np

from brainmri.core.mri_slice import MRISlice


def volume_ml(slices: List[MRISlice], voxel_size: float, intensity=None) -> float:

    total_volume_ml = 0

    for slice in slices:
        total_volume_ml += volume_ml_per_slice(slice, voxel_size, intensity=intensity)
    return total_volume_ml


def volume_ml_per_slice(slice: MRISlice, voxel_size: float, intensity=None) -> float:

    if intensity == 0:
        raise ValueError("Intensity must be greater than 0")

    if intensity is None:
        mask = slice.data > 0
    else:
        mask = slice.data == intensity
    slice_voxels = np.count_nonzero(mask)
    slice_volume = slice_voxels * voxel_size
    return slice_volume / 1000
