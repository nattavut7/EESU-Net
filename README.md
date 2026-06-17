# EESU-Net

Official PyTorch implementation template for:

**EESU-Net: Edge-Enhanced Swin-Transformer Network for Accurate and Efficient 3D Brain-Tumor Segmentation**

Published in **Neural Computing and Applications**, Springer Nature, 2026.  
DOI: https://doi.org/10.1007/s00521-026-12283-w

## Highlights

- Swin-Transformer encoder
- Edge-Enhancement Module (EEM)
- Lightweight Convolutional Attention (LCA) decoder
- Hybrid Dice-Boundary loss
- BraTS2021 benchmark

## Reported Results

| Metric | Value |
|---|---:|
| Dice | 96.2 ± 0.3% |
| IoU | 90.8 ± 0.2% |
| HD95 | 3.8 ± 0.2 mm |
| Parameters | ≈19.6 M |
| Inference time | ≈1.3 s per MRI volume |

## Installation

```bash
conda env create -f environment.yml
conda activate eesunet
```

or

```bash
pip install -r requirements.txt
```

## Training

```bash
python train.py --config configs/eesunet_brats2021.yaml
```

## Testing

```bash
python test.py --config configs/eesunet_brats2021.yaml --checkpoint pretrained/eesunet_brats2021.pth
```

## Inference

```bash
python inference.py --input /path/to/case --checkpoint pretrained/eesunet_brats2021.pth --output outputs/
```

## Dataset

The BraTS2021 dataset is not included. Please download it from the official BraTS/Kaggle/Synapse source according to its license.

Expected case format:

```text
BraTS2021_00000/
├── flair.nii.gz
├── t1.nii.gz
├── t1ce.nii.gz
├── t2.nii.gz
└── seg.nii.gz
```

## Citation

```bibtex
@article{sriwiboon2026eesunet,
  title={EESU-Net: edge-enhanced swin-transformer network for accurate and efficient 3D brain-tumor segmentation},
  author={Sriwiboon, Nattavut},
  journal={Neural Computing and Applications},
  volume={38},
  pages={467},
  year={2026},
  doi={10.1007/s00521-026-12283-w}
}
```
