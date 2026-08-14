# Vendored (verbatim) from the user's own PNP_segmentation-main project,
# utils.py -- only CLASS_NEIGHBORS, LABELS, and build_class_weights are kept
# (the pieces admm_x_update_segmentation needs); that file's dataset/FID/
# plotting utilities aren't relevant here.

import torch

CLASS_NEIGHBORS = {
    0:  [1, 13, 14, 17, 18],       # background
    1:  [2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 17],  # skin
    2:  [1, 4, 5, 10],             # nose
    3:  [1, 4, 5, 6, 7],           # eye_g
    4:  [1, 2, 3, 6],              # l_eye
    5:  [1, 2, 3, 7],              # r_eye
    6:  [1, 4, 13],                # l_brow
    7:  [1, 5, 13],                # r_brow
    8:  [1, 13, 15],               # l_ear
    9:  [1, 13, 15],               # r_ear
    10: [1, 2, 11, 12],            # mouth
    11: [1, 10, 12],               # u_lip
    12: [1, 10, 11],               # l_lip
    13: [0, 1, 6, 7, 8, 9, 14],   # hair
    14: [0, 13],                   # hat
    15: [8, 9, 13],                # ear_r
    16: [1, 17, 18],               # neck_l
    17: [0, 1, 16, 18],            # neck
    18: [0, 16, 17],               # cloth
}

LABELS = [
    "background", "skin", "nose", "eye_g", "l_eye", "r_eye",
    "l_brow", "r_brow", "l_ear", "r_ear", "mouth", "u_lip",
    "l_lip", "hair", "hat", "ear_r", "neck_l", "neck", "cloth"
]


def build_class_weights(num_classes: int = 19) -> torch.Tensor:
    weights = {
        "background": 0.4665, "skin": 0.4785, "nose": 0.7903, "eye_g": 1.1833,
        "l_eye": 1.2324, "r_eye": 1.2332, "l_brow": 1.0854, "r_brow": 1.0907,
        "l_ear": 1.0655, "r_ear": 1.1019, "mouth": 1.1601, "u_lip": 1.0886,
        "l_lip": 0.9862, "hair": 0.4581, "hat": 0.9275, "ear_r": 1.2117,
        "neck_l": 2.0371, "neck": 0.6874, "cloth": 0.7155
    }
    w = torch.ones(num_classes)
    for idx, name in enumerate(LABELS):
        w[idx] = weights.get(name, 1.0)
    return w
