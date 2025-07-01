import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models, utils
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Dataset class
def get_bbox_from_mask(mask_path):
    mask = Image.open(mask_path).convert("L")
    mask = np.array(mask)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [xs.min(), ys.min(), xs.max(), ys.max()]

class BoatYawDatasetResNet(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)],
                p=0.3),
            transforms.GaussianBlur(kernel_size=3),
            transforms.RandomRotation(degrees=5),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.5)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.samples = []

        for filename in os.listdir(data_dir):
            if not filename.endswith(".main.json"):
                continue

            base = filename.replace(".main.json", "")
            json_path = os.path.join(data_dir, f"{base}.main.json")
            img_path = os.path.join(data_dir, f"{base}.png")
            mask_path = os.path.join(data_dir, f"{base}_seg.png")

            if not os.path.exists(img_path) or not os.path.exists(mask_path):
                continue

            with open(json_path, 'r') as f:
                data = json.load(f)

            for obj in data.get("objects", []):
                if not obj.get("bounding_valid", False):
                    continue

                bbox = get_bbox_from_mask(mask_path)
                if bbox is None:
                    continue

                yaw = obj["relative_rotation"][1]
                self.samples.append({
                    "img_path": img_path,
                    "bbox": bbox,
                    "yaw": yaw
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        image = Image.open(item["img_path"]).convert("RGB")
        x1, y1, x2, y2 = item["bbox"]
        cropped = image.crop((x1, y1, x2, y2))
        cropped = self.transform(cropped)
        yaw = torch.tensor(item["yaw"], dtype=torch.float32)
        return cropped, yaw

# Training script
def train(args):
    data_dir = args.data_dir

    dataset = BoatYawDatasetResNet(data_dir)
    val_ratio = 0.2
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size

    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    writer = SummaryWriter(log_dir="runs/yaw_regression")

    num_epochs = args.num_epochs
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = model(x).squeeze(1)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
        epoch_train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        running_val_loss = 0.0
        running_mae = 0.0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                y_pred = model(x_val).squeeze(1)
                loss = criterion(y_pred, y_val)
                running_val_loss += loss.item() * x_val.size(0)
                running_mae += torch.sum(torch.abs(y_pred - y_val)).item()
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        epoch_val_mae = running_mae / len(val_loader.dataset)

        writer.add_scalar("Loss/Train", epoch_train_loss, epoch)
        writer.add_scalar("Loss/Val", epoch_val_loss, epoch)
        writer.add_scalar("MAE/Val", epoch_val_mae, epoch)

        if epoch == 0:
            x_val_cpu = x_val.cpu()
            img_grid = utils.make_grid(x_val_cpu)
            writer.add_image("val_images", img_grid, epoch)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), "best_resnet_yaw.pth")

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_train_loss:.2f} | Val Loss: {epoch_val_loss:.2f} | Val MAE: {epoch_val_mae:.2f}°")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a ResNet model for yaw angle regression.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and validation.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for the optimizer.")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs for training.")

    args = parser.parse_args()
    train(args)
