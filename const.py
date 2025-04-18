import enum
import os

import numpy as np
from pathlib import Path

# tooth existence of the patient: FDI numbering order 11-17, 21-27, 31-37, 41-47
TOOTH_EXIST_MASK = {"0": np.ones((28,), np.bool_), "1": np.ones((28,), np.bool_)}


# Mask used to project the contour of the selected teeth in photos of different views
MASK_UPPER = np.array(
    [
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
)

MASK_LOWER = np.array(
    [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
    ]
)

MASK_LEFT = np.array(
    [
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
)

MASK_RIGHT = np.array(
    [
        True,
        True,
        True,
        True,
        True,
        True,
        False,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
    ]
)

MASK_FRONTAL = np.array(
    [
        True,
        True,
        True,
        True,
        True,
        True,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
    ]
)


@enum.unique
class PHOTO(enum.Enum):
    # Enum values must be 0,1,2,3,4
    UPPER = 0
    LOWER = 1
    LEFT = 2
    RIGHT = 3
    FRONTAL = 4


PHOTO_TYPES = [PHOTO.UPPER, PHOTO.LOWER, PHOTO.LEFT, PHOTO.RIGHT, PHOTO.FRONTAL]
VISIBLE_MASKS = [MASK_UPPER, MASK_LOWER, MASK_LEFT, MASK_RIGHT, MASK_FRONTAL]
RECONS_IMG_WIDTH = 800

PHOTO_DIR = r"./seg/valid/image"
PHOTO_DIR = Path(PHOTO_DIR)

NUM_PC = 10  # num of modes of deformation for each tooth used in reconstruction
NUM_POINT = 1500  # num of points to represent tooth surface used in SSM
PG_SHAPE = (NUM_POINT, 3)

# FDI TOOTH NUMEBRING
UPPER_INDICES = [
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
]  # ignore wisdom teeth
LOWER_INDICES = [
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
]  # ignore wisdom teeth

SSM_DIR = r"./ssm/eigValVec/"
REGIS_PARAM_DIR = r"./ssm/cpdGpParams/"
DEMO_H5_DIR = r"./demo/h5/"
DEMO_MESH_DIR = r"./demo/mesh/"
REF_MESH_DIR = r"./demo/ref_mesh/"
VIS_DIR = r"./demo/visualization"

SSM_DIR = Path(SSM_DIR).resolve()
REGIS_PARAM_DIR = Path(REGIS_PARAM_DIR).resolve()
DEMO_H5_DIR = Path(DEMO_H5_DIR).resolve()
DEMO_MESH_DIR = Path(DEMO_MESH_DIR).resolve()
REF_MESH_DIR = Path(REF_MESH_DIR).resolve()
VIS_DIR = Path(VIS_DIR).resolve()

os.makedirs(DEMO_H5_DIR, exist_ok=True)
os.makedirs(DEMO_MESH_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)
