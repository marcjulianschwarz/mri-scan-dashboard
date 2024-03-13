import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from brainmri.constants import MODELS_PATH
from brainmri.core import MRIImage


def get_start_end_slice_nr(mri_image: MRIImage):
    start = 0
    for i, slice in enumerate(mri_image.slices_z):
        if np.sum(slice.data) > 0:
            start = i
            break

    end = 0
    for i, slice in enumerate(mri_image.slices_z[::-1]):
        if np.sum(slice.data) > 0:
            end = len(mri_image.slices_z) - i
            break
    return start, end


def get_slice_nrs_through_mri(mri_image: MRIImage, k=5):
    k = k + 2
    start, end = get_start_end_slice_nr(mri_image)
    end = min(end, len(mri_image.slices_z) - 1)  # ensure end does not exceed the length
    slice_nrs = np.linspace(start, end, num=k, endpoint=False).astype(
        int
    )  # Generate k numbers from start to end, not including end
    return slice_nrs.tolist()[1:-1]


# def get_features(mri_image):
#     slice_nrs = get_slice_nrs_through_mri(mri_image)
#     features = []
#     for slice_nr in slice_nrs:
#         slice = mri_image.slices_z[slice_nr]
#         if np.sum(slice) == 0:
#             continue
#         cropped_slice = crop_image(slice)
#         padded_slice = pad_image_center(cropped_slice)
#         normalized_slice = normalize_image(padded_slice)
#         features.append(normalized_slice)
#     return features
# def get_feature_overview(mri_image):
#     slice_nrs = get_slice_nrs_through_mri(mri_image)
#     print(slice_nrs)
#     fig, axs = plt.subplots(1, len(slice_nrs), figsize=(15, 5))
#     for i, slice_nr in enumerate(slice_nrs):
#         img = crop_image(mri_image.slices_z[slice_nr])
#         img = pad_image_center(img)
#         img = normalize_image(img)
#         axs[i].imshow(img, cmap="gray")
#         axs[i].axis("off")


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        in_channels = 1
        out_channels = 10
        kernel_size = 5

        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.conv2 = nn.Conv2d(10, 20, 3)
        self.conv3 = nn.Conv2d(20, 30, 4)
        self.conv4 = nn.Conv2d(30, 40, 5)

        self.fc1 = nn.Linear(40 * 12 * 12, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 100)
        self.fc4 = nn.Linear(100, 10)
        self.fc5 = nn.Linear(10, 1)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))

        # print(x.shape)
        x = x.flatten(1)
        x = F.tanh(x)
        x = self.fc1(x)
        x = F.tanh(x)
        x = self.fc2(x)
        x = F.tanh(x)
        x = self.fc3(x)
        x = F.tanh(x)
        x = self.fc4(x)
        x = self.fc5(x)
        # print(x.shape)
        return x


def predict_ga(mri_image: MRIImage):
    net = torch.load(MODELS_PATH / "net.pth")
    slice = mri_image.max_size_slice()
    slice = slice.crop().pad(256).torch_tensor()
    slice = slice.unsqueeze(0)  # add channel dimension
    slice = (slice - slice.mean()) / slice.std()
    slice = slice.unsqueeze(0)  # add batch dimension
    print(slice.shape)

    with torch.no_grad():
        output = net(slice)
        ga = output.item()

    return ga


def predict_ga_reg(mri_image: MRIImage):
    reg = pickle.load(open(MODELS_PATH / "reg.pkl", "rb"))
    # ga = mri_image.gestational_age()
    vol = mri_image.volume_ml()
    hc = mri_image.head_circumference()
    X = [vol, hc]
    y_pred = reg.predict([X])
    return y_pred[0]
