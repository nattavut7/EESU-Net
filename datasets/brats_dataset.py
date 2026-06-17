from pathlib import Path
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


class BraTSDataset(Dataset):
    def __init__(self, root, cases=None, modalities=("flair", "t1", "t1ce", "t2"), transform=None):
        self.root = Path(root)
        self.modalities = modalities
        self.transform = transform
        self.cases = cases if cases is not None else sorted([p for p in self.root.iterdir() if p.is_dir()])

    def __len__(self):
        return len(self.cases)

    def _load_nii(self, path):
        return nib.load(str(path)).get_fdata().astype(np.float32)

    def __getitem__(self, idx):
        case_dir = Path(self.cases[idx])
        images = []
        for m in self.modalities:
            candidates = list(case_dir.glob(f"*{m}*.nii.gz"))
            if not candidates:
                raise FileNotFoundError(f"Missing modality {m} in {case_dir}")
            images.append(self._load_nii(candidates[0]))

        x = np.stack(images, axis=0)
        x = (x - x.mean(axis=(1, 2, 3), keepdims=True)) / (x.std(axis=(1, 2, 3), keepdims=True) + 1e-8)

        seg_candidates = list(case_dir.glob("*seg*.nii.gz"))
        if seg_candidates:
            y = self._load_nii(seg_candidates[0]).astype(np.int64)
            y[y == 4] = 3
        else:
            y = np.zeros(x.shape[1:], dtype=np.int64)

        sample = {"image": x, "label": y, "case": case_dir.name}
        if self.transform:
            sample = self.transform(sample)
        return torch.from_numpy(sample["image"]).float(), torch.from_numpy(sample["label"]).long()
