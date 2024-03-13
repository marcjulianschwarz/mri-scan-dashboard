import numpy as np
import torch


class Image:

    data: np.ndarray

    def __init__(self, data: np.ndarray) -> None:
        self.data = data

    def crop(self):
        self.data = crop_image(self.data)
        return self

    def pad(self, size=200):
        self.data = pad_image_center(self.data, size=size)
        return self

    def normalize(self):
        self.data = normalize_image(self.data)
        return self

    def torch_tensor(self):
        return torch.tensor(self.data).float()

    def add_channel_dim(self):
        return self.data.unsqueeze(0)


def crop_image(image):
    mask = image == 0
    coords = np.array(np.nonzero(~mask))
    top_left = np.min(coords, axis=1)
    bottom_right = np.max(coords, axis=1)
    cropped_image = image[top_left[0] : bottom_right[0], top_left[1] : bottom_right[1]]
    return cropped_image


def pad_image_center(image, size=200):
    height, width = image.shape

    # Calculate padding for height and width
    pad_height = size - height
    pad_width = size - width

    # Distribute the padding equally to top/bottom and left/right, but if the padding is odd, add the extra to the bottom/right
    pad_top = pad_height // 2
    pad_bottom = pad_height // 2 + pad_height % 2
    pad_left = pad_width // 2
    pad_right = pad_width // 2 + pad_width % 2

    padded_image = np.pad(
        image, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant"
    )

    return padded_image


def normalize_image(image):
    min_val = np.min(image)
    max_val = np.max(image)

    if max_val - min_val == 0:  # Check if the denominator is zero
        return np.zeros_like(image)
    else:
        return (image - min_val) / (max_val - min_val)
