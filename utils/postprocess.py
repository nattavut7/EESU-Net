import numpy as np
from scipy import ndimage


def keep_largest_component(mask):
    labeled, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    largest = int(np.argmax(sizes)) + 1
    return labeled == largest


def postprocess_prediction(pred):
    out = pred.copy()
    for cls in [1, 2, 3]:
        out_cls = keep_largest_component(out == cls)
        out[(out == cls) & (~out_cls)] = 0
    return out
