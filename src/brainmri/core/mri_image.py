from pathlib import Path
from typing import List

import nibabel as nib
import numpy as np

from .gestational_age import gestational_age
from .head_circum import head_circumference
from .intensity import intensity_values
from .mri_slice import MRISlice
from .volume import volume_ml


class MRIImage:
    filename: str
    study: str
    id_nr: int
    data: np.ndarray
    zooms: tuple
    voxel_size: float

    slices_x: List[MRISlice]
    slices_y: List[MRISlice]
    slices_z: List[MRISlice]

    max_slice_nr_x: int
    max_slice_nr_y: int
    max_slice_nr_z: int

    def gestational_age(self):
        return gestational_age_mri(self)

    def max_slice_nr(self, axis: str):
        return max_slice_nr_mri(self, axis)

    def intensity_values(self):
        return intensity_values_mri(self)

    def volume_ml(self, intensity=None):
        return volume_ml_mri(self, intensity=intensity)

    def head_circumference(self):
        return head_circumference_mri(self)

    def max_size_slice(self) -> MRISlice:
        return max_size_slice(self)

    def slice(self, axis: str, slice_nr: int) -> MRISlice:
        return get_slice(self, axis, slice_nr)

    @classmethod
    def from_file(cls, path: Path):
        return mri_image_from_file(path)


def mri_image_from_file(path: Path):

    mri_image = MRIImage()

    filename = path.stem
    study = filename.split("_")[0]
    study2 = study.split("fm")[1:]

    mri_image.filename = filename
    mri_image.study = study
    mri_image.id_nr = int(study2[0])

    mri_file = nib.load(path)
    zooms = mri_file.header.get_zooms()
    voxel_size = np.prod(zooms)

    mri_image.zooms = zooms
    mri_image.voxel_size = voxel_size

    data = mri_file.get_fdata()
    mri_image.data = data

    nx, ny, nz = data.shape
    # flip left right and rotate 90 degrees to get the correct orientation
    mri_image.slices_x = [
        MRISlice(i, "x", np.fliplr(np.rot90(data[i, :, :], 1))) for i in range(nx)
    ]
    mri_image.slices_y = [
        MRISlice(i, "y", np.fliplr(np.rot90(data[:, i, :], 1))) for i in range(ny)
    ]
    mri_image.slices_z = [
        MRISlice(i, "z", np.fliplr(np.rot90(data[:, :, i], 1))) for i in range(nz)
    ]

    mri_image.max_slice_nr_x = nx - 1
    mri_image.max_slice_nr_y = ny - 1
    mri_image.max_slice_nr_z = nz - 1

    return mri_image


def max_slice_nr(mri_image: MRIImage, axis: str) -> int:
    if axis == "x":
        return mri_image.max_slice_nr_x
    elif axis == "y":
        return mri_image.max_slice_nr_y
    elif axis == "z":
        return mri_image.max_slice_nr_z
    else:
        raise ValueError("Axis must be 'x', 'y' or 'z'")


def max_slice_nr_mri(mri_image: MRIImage, axis: str):
    return max_slice_nr(mri_image, axis)


def intensity_values_mri(mri_image: MRIImage):
    return intensity_values(mri_image.data)


def gestational_age_mri(mri_image: MRIImage):
    return gestational_age(mri_image.id_nr)


def volume_ml_mri(mri_image: MRIImage, intensity=None):
    return volume_ml(mri_image.slices_z, mri_image.voxel_size, intensity=intensity)


def head_circumference_mri(mri_image: MRIImage):
    pixel_size_mm = mri_image.zooms[-1]
    slices = mri_image.slices_z[20:-20]
    return head_circumference(pixel_size_mm, slices)


def max_size_slice(mri_image: MRIImage) -> MRISlice:
    max_total = 0
    max_slice = None
    for slice in mri_image.slices_z:
        mask = slice.data == 0
        coords = np.array(np.nonzero(~mask))
        if coords.size == 0:
            continue
        bottom_y, bottom_x = np.max(coords, axis=1)
        top_y, top_x = np.min(coords, axis=1)
        height = bottom_y - top_y
        width = bottom_x - top_x
        total = height + width
        if total > max_total:
            max_total = total
            max_slice = slice
    return max_slice


def get_slice(mir_image: MRIImage, axis: str, slice_nr: int) -> MRISlice:
    if axis == "x":
        return mir_image.slices_x[slice_nr]
    elif axis == "y":
        return mir_image.slices_y[slice_nr]
    elif axis == "z":
        return mir_image.slices_z[slice_nr]
    else:
        raise ValueError("Axis must be 'x', 'y' or 'z'")
