import random
import numpy as np


class RandomCrop3D:
    def __init__(self, size=(128, 128, 128)):
        self.size = size

    def __call__(self, sample):
        x, y = sample["image"], sample["label"]
        d, h, w = y.shape
        cd, ch, cw = self.size
        sd = random.randint(0, max(d - cd, 0))
        sh = random.randint(0, max(h - ch, 0))
        sw = random.randint(0, max(w - cw, 0))
        sample["image"] = x[:, sd:sd+cd, sh:sh+ch, sw:sw+cw]
        sample["label"] = y[sd:sd+cd, sh:sh+ch, sw:sw+cw]
        return sample


class RandomFlip3D:
    def __call__(self, sample):
        x, y = sample["image"], sample["label"]
        for axis in [1, 2, 3]:
            if random.random() < 0.5:
                x = np.flip(x, axis=axis).copy()
                y = np.flip(y, axis=axis - 1).copy()
        sample["image"], sample["label"] = x, y
        return sample


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for transform in self.transforms:
            sample = transform(sample)
        return sample
