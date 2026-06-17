import argparse
from pathlib import Path
import torch
import yaml
import nibabel as nib
import numpy as np
from models.eesunet import EESUNet


def load_case(case_dir, modalities=("flair", "t1", "t1ce", "t2")):
    case_dir = Path(case_dir)
    imgs = []
    affine = None
    for m in modalities:
        path = list(case_dir.glob(f"*{m}*.nii.gz"))[0]
        nii = nib.load(str(path))
        affine = nii.affine
        data = nii.get_fdata().astype(np.float32)
        data = (data - data.mean()) / (data.std() + 1e-8)
        imgs.append(data)
    return np.stack(imgs, axis=0), affine


def main(config, checkpoint, input_dir, output_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EESUNet(
        in_channels=4,
        out_channels=config["dataset"]["num_classes"],
        base_channels=config["model"]["base_channels"],
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    x, affine = load_case(input_dir, config["dataset"]["modalities"])
    x_tensor = torch.from_numpy(x[None]).float().to(device)

    with torch.no_grad():
        pred = torch.argmax(model(x_tensor), dim=1).cpu().numpy()[0].astype(np.uint8)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir) / "eesunet_prediction.nii.gz"
    nib.save(nib.Nifti1Image(pred, affine), str(out_path))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eesunet_brats2021.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="outputs")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    main(cfg, args.checkpoint, args.input, args.output)
