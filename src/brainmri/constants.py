import os
from pathlib import Path

import pandas as pd

DATA_PATH = Path(os.environ.get("DATA_PATH"))
DATA_TASK_PATH = DATA_PATH / "data_task_2"
MODELS_PATH = Path(os.environ.get("MODELS_PATH"))
IMAGES_PATH = Path(os.environ.get("IMAGES_PATH"))
PLOTS_PATH = Path(os.environ.get("PLOTS_PATH"))

img_paths = list(DATA_TASK_PATH.glob("*.nii.gz"))

brain_image_paths = [img for img in img_paths if "brain" in img.stem]
other_image_paths = [img for img in img_paths if not "brain" in img.stem]
BRAIN_IMAGE_NAME_TO_PATH_MAPPING = {img.stem: img for img in brain_image_paths}

gast_df = pd.read_csv(str(DATA_TASK_PATH / "gestational_ages.csv"))
GESTATIONAL_AGE_IDS = list(gast_df["ids"])
GESTATIONAL_AGES = list(gast_df["tag_ga"])

INTENSITY_TO_NAME_MAPPING = {
    "0": "eCSF_L",
    "1": "eCSF_R",
    "2": "Cortex_L",
    "3": "Cortex_R",
    "4": "WM_L",
    "5": "WM_R",
    "6": "Lat_ve ntricle_L",
    "7": "Lat_ventricle_R",
    "8": "CSP",
    "9": "Brainstem",
    "10": "Cerebellum_L",
    "11": "Cerebellum_R",
    "12": "Vermis",
    "13": "Lentiform_L",
    "14": "Lentiform_R",
    "15": "Thalamus_L",
    "16": "Thalamus_R",
    "17": "Third_ventricle",
    "18": "?",
    "19": "?",
}
