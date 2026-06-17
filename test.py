import argparse
import yaml
import torch
from models.eesunet import EESUNet


def main(config, checkpoint):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EESUNet(
        in_channels=4,
        out_channels=config["dataset"]["num_classes"],
        base_channels=config["model"]["base_channels"],
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    print("Model loaded. Customize test.py with your BraTS split for full evaluation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eesunet_brats2021.yaml")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    main(cfg, args.checkpoint)
