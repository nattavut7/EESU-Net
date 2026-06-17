import argparse
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader

from models.eesunet import EESUNet
from models.losses import DiceBoundaryLoss
from datasets.brats_dataset import BraTSDataset
from datasets.transforms import Compose, RandomCrop3D, RandomFlip3D


def train(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = Compose([RandomCrop3D(tuple(config["training"]["input_size"])), RandomFlip3D()])
    dataset = BraTSDataset(config["dataset"]["root"], transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    model = EESUNet(
        in_channels=4,
        out_channels=config["dataset"]["num_classes"],
        base_channels=config["model"]["base_channels"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"]["learning_rate"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5)
    criterion = DiceBoundaryLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=config["training"].get("mixed_precision", True))

    for epoch in range(int(config["training"]["epochs"])):
        model.train()
        running_loss = 0.0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=config["training"].get("mixed_precision", True)):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        epoch_loss = running_loss / max(len(loader), 1)
        scheduler.step(epoch_loss)
        print(f"Epoch {epoch + 1:03d} | loss={epoch_loss:.4f}")

    Path("pretrained").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "pretrained/eesunet_brats2021.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eesunet_brats2021.yaml")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    train(cfg)
