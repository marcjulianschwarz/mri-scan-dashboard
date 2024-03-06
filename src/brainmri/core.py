import os
from pathlib import Path

import cv2 as cv
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

DATA_PATH = Path(os.environ.get("DATA_PATH"))
DATA_TASK_PATH = DATA_PATH / "data_task_2"
MODELS_PATH = Path(os.environ.get("MODELS_PATH"))
IMAGES_PATH = Path(os.environ.get("IMAGES_PATH"))
PLOTS_PATH = Path(os.environ.get("PLOTS_PATH"))

img_paths = list(DATA_TASK_PATH.glob("*.nii.gz"))
print("Loading Images")

brain_image_paths = [img for img in img_paths if "brain" in img.stem]
other_image_paths = [img for img in img_paths if not "brain" in img.stem]
brain_image_name_to_path = {img.stem: img for img in brain_image_paths}

gest_df = pd.read_csv(str(DATA_TASK_PATH / "gestational_ages.csv"))
gas_ids = list(gest_df["ids"])
gas = list(gest_df["tag_ga"])

intensity_to_name = {}
intensity_to_name["0"] = "eCSF_L"
intensity_to_name["1"] = "eCSF_R"
intensity_to_name["2"] = "Cortex_L"
intensity_to_name["3"] = "Cortex_R"
intensity_to_name["4"] = "WM_L"
intensity_to_name["5"] = "WM_R"
intensity_to_name["6"] = "Lat_ventricle_L"
intensity_to_name["7"] = "Lat_ventricle_R"
intensity_to_name["8"] = "CSP"
intensity_to_name["9"] = "Brainstem"
intensity_to_name["10"] = "Cerebellum_L"
intensity_to_name["11"] = "Cerebellum_R"
intensity_to_name["12"] = "Vermis"
intensity_to_name["13"] = "Lentiform_L"
intensity_to_name["14"] = "Lentiform_R"
intensity_to_name["15"] = "Thalamus_L"
intensity_to_name["16"] = "Thalamus_R"
intensity_to_name["17"] = "Third_ventricle"
intensity_to_name["18"] = "?"
intensity_to_name["19"] = "?"


class MRIImage:

    filename: str
    study: str
    id_nr: int
    ga: float
    zooms: tuple

    slices_x: list
    slices_y: list
    slices_z: list

    max_slice_nr_x: int
    max_slice_nr_y: int
    max_slice_nr_z: int

    voxel_size: float

    intensities: list

    @classmethod
    def from_file(cls, path: Path):
        self = cls()

        self.filename = path.stem
        self.study = self.filename.split("_")[0]
        self.study2 = self.study.split("fm")[1:]
        self.id_nr = int(self.study2[0])
        self.ga = self.get_ga(self.id_nr)

        self.mri_file = nib.load(path)
        self.zooms = self.mri_file.header.get_zooms()
        self.voxel_size = np.prod(self.zooms)

        data = self.mri_file.get_fdata()

        nx, ny, nz = data.shape
        # flip left right and rotate 90 degrees to get the correct orientation
        self.slices_x = [np.fliplr(np.rot90(data[i, :, :], 1)) for i in range(nx)]
        self.slices_y = [np.fliplr(np.rot90(data[:, i, :], 1)) for i in range(ny)]
        self.slices_z = [np.fliplr(np.rot90(data[:, :, i], 1)) for i in range(nz)]

        self.max_slice_nr_x = nx - 1
        self.max_slice_nr_y = ny - 1
        self.max_slice_nr_z = nz - 1

        self.intensities = get_possible_intensity_values(data)

        return self

    def get_max_slice_nr(self, axis: str):
        if axis == "x":
            return self.max_slice_nr_x
        elif axis == "y":
            return self.max_slice_nr_y
        elif axis == "z":
            return self.max_slice_nr_z
        else:
            raise ValueError("Axis must be 'x', 'y' or 'z'")

    def get_ga(self, id_nr: int):

        if id_nr in gas_ids:
            index_in_list = gas_ids.index(id_nr)
            ga = gas[index_in_list]
            if type(ga) == int or type(ga) == float and not np.isnan(ga):
                ga = ga
            else:
                ga = None
        else:
            ga = 0
        return ga


def get_slice(mir_image: MRIImage, axis: str, slice_nr: int):
    if axis == "x":
        return mir_image.slices_x[slice_nr]
    elif axis == "y":
        return mir_image.slices_y[slice_nr]
    elif axis == "z":
        return mir_image.slices_z[slice_nr]
    else:
        raise ValueError("Axis must be 'x', 'y' or 'z'")


def get_slices(mir_image: MRIImage, axis: str):
    if axis == "x":
        return mir_image.slices_x
    elif axis == "y":
        return mir_image.slices_y
    elif axis == "z":
        return mir_image.slices_z
    else:
        raise ValueError("Axis must be 'x', 'y' or 'z'")


def plot_slice(slice, return_fig=False):

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(slice, cmap="gray")
    if return_fig:
        return fig


def get_total_volume_in_ml(mri_image: MRIImage, intensity=None):

    total_volume_ml = 0
    # choosing the z axis as it makes no difference which axis we choose
    for slice in mri_image.slices_z:
        total_volume_ml += get_volume_in_ml_per_slice(
            slice, mri_image.voxel_size, intensity=intensity
        )
    return total_volume_ml


def get_volume_in_ml_per_slice(slice, voxel_size, intensity=None):

    if intensity == 0:
        raise ValueError("Intensity must be greater than 0")

    if intensity is None:
        mask = slice > 0
    else:
        mask = slice == intensity
    slice_voxels = np.count_nonzero(mask)
    slice_volume = slice_voxels * voxel_size
    return slice_volume / 1000


def get_possible_intensity_values(img):
    intensities = np.unique(img)
    intensities = intensities[intensities > 0]
    return intensities


def get_mask(slice, intensity=None):
    if intensity is None:
        mask = slice > 0
    else:
        mask = slice == intensity
    return mask


def get_head_circumference(mri_image: MRIImage):
    pixel_size_mm = mri_image.zooms[-1]
    max_c = 0
    max_img = None
    for img in mri_image.slices_z[20:-20]:
        mask = img == 0
        coords = np.array(np.nonzero(~mask))
        if coords.size == 0:
            continue
        bottom_y, bottom_x = np.max(coords, axis=1)
        top_y, top_x = np.min(coords, axis=1)
        a = (bottom_y - top_y) * pixel_size_mm
        b = (bottom_x - top_x) * pixel_size_mm
        cirum_eq = 1.62 * (a + b) / 10

        img = get_mask(img)
        img = np.array(img, dtype=np.uint8)
        img = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX)

        contours, hierarchy = cv.findContours(
            img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )
        blank_img = np.zeros_like(img)
        img_with_contours = cv.drawContours(blank_img, contours, -1, 255, 1)

        circum_im = sum(img_with_contours.flatten() == 255) * pixel_size_mm / 10

        avg = (cirum_eq + circum_im) / 2
        if avg > max_c:
            max_c = avg
            max_img = img

    return max_c


def get_max_slice(mri_image):
    max_total = 0
    max_slice = None
    for slice in mri_image.slices_z:
        mask = slice == 0
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


def plot_3d_brain(mri_image: MRIImage, to_slide_nr: int = None):

    # Assuming slices is a list of 2D numpy arrays
    slices = mri_image.slices_z[::2]

    stack = []

    for i, slice in enumerate(slices[:to_slide_nr]):
        x, y = np.where(slice > 0)  # retain this to limit to non-zero elements
        z = [i * 2] * len(x)
        intensity = slice[x, y]  # get the intensity values
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


def plot_animator(slices):
    fig = px.imshow(np.array(slices), animation_frame=0, binary_string=True)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(autosize=False, width=500, height=500, coloraxis_showscale=False)
    # Drop animation buttons
    fig["layout"].pop("updatemenus")
    return fig
