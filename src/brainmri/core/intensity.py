import numpy as np


def intensity_values(img: np.ndarray):
    intensities = np.unique(img)
    intensities = intensities[intensities > 0]
    return intensities
