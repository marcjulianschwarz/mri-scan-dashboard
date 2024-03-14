from typing import List

import cv2 as cv
import numpy as np

from .mri_slice import MRISlice


def get_mask(image: np.ndarray, intensity=None):
    if intensity is None:
        mask = image > 0
    else:
        mask = image == intensity
    return mask


def head_circumference(pixel_size_mm: float, slices: List[MRISlice]) -> float:

    max_c = 0

    # find maximum head circumference
    for slice in slices:
        image = slice.data
        # crop image
        coords = np.array(np.nonzero(~(image == 0)))
        if coords.size == 0:
            continue
        bottom_y, bottom_x = np.max(coords, axis=1)
        top_y, top_x = np.min(coords, axis=1)

        # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5504265/
        # Kyriakopoulou V, Vatansever D, Davidson A, et al. Normative biometry of the
        # fetal brain using magnetic resonance imaging. Brain Struct Funct.
        # 2017;222(5):2295-2307. doi:10.1007/s00429-016-1342-6
        # 1.62 × [(skull biparietal diameter) + (skull occipitofrontal diameter)]
        skull_occipitofrontal_diameter = (bottom_y - top_y) * pixel_size_mm
        skull_biparietal_diameter = (bottom_x - top_x) * pixel_size_mm
        circum_equ_cm = (
            1.62 * (skull_occipitofrontal_diameter + skull_biparietal_diameter) / 10
        )

        image = get_mask(image)
        image = np.array(image, dtype=np.uint8)
        image = cv.normalize(image, None, 0, 255, cv.NORM_MINMAX)

        # Use contours to find circumference
        contours, hierarchy = cv.findContours(
            image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )
        blank_image = np.zeros_like(image)
        image_with_contours = cv.drawContours(blank_image, contours, -1, 255, 1)

        circum_im_cm = sum(image_with_contours.flatten() == 255) * pixel_size_mm / 10

        avg = (circum_equ_cm + circum_im_cm) / 2
        if avg > max_c:
            max_c = avg

    return max_c
