"""
Task 2: R-CNN Model Architecture & Multi-Task Training
======================================================
Author: Muhammad John Raza (23k-0069)
FAST NUCES Karachi - Department of Artificial Intelligence
Course: Computer Vision Lab (Lab 08)

R-CNN Step 2:
- Loads warped proposal crops (224x224) and ground-truth labels/deltas.
- Builds CNN Feature Extractor (ResNet18 backbone) + Multi-task Heads:
  1. Softmax Object Classifier (Classes: Background, Car, Pedestrian, Traffic_Sign)
  2. Bounding Box Regressor (Refines tx, ty, tw, th offsets)
- Trains with CrossEntropyLoss and SmoothL1Loss.
- Saves model weights to outputs/rcnn_model.pth.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms

def get_path(rel_path):
    if os.path.exists(rel_path):
        return rel_path
    if rel_path.startswith("lab8/") and os.path.exists(rel_path[5:]):
        return rel_path[5:]
    if os.path.exists(os.path.join("lab8", rel_path)):
        return os.path.join("lab8", rel_path)
    if os.path.basename(os.getcwd()) == "lab8" and rel_path.startswith("lab8/"):
        return rel_path[5:]
    return rel_path

output_dir = "outputs" if os.path.basename(os.getcwd()) == "lab8" else "lab8/outputs"
os.makedirs(output_dir, exist_ok=True)

class RCNNDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.crops = data['crops'] # N, 224, 224, 3 (BGR uint8)
        self.labels = data['labels'] # N (int64)
        self.deltas = data['deltas'] # N, 4 (float32)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, idx):
        crop = self.crops[idx] # BGR
        crop_rgb = crop[:, :, ::-1].copy() # RGB
        tensor_crop = self.transform(crop_rgb)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        delta = torch.tensor(self.deltas[idx], dtype=torch.float32)
        return tensor_crop, label, delta

class RCNNModel(nn.Module):
    def __init__(self, num_classes=4):
        super(RCNNModel, self).__init__()
        # Pretrained ResNet18 backbone
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Freeze early layers for fast and stable transfer learning
        for param in list(backbone.parameters())[:-12]:
            param.requires_grad = False

        self.features = nn.Sequential(*list(backbone.children())[:-1]) # Output: 512 x 1 x 1
        in_features = 512

        # 1. Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

        # 2. Bounding Box Regression Head
        self.bbox_regressor = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4) # (tx, ty, tw, th)
        )

    def forward(self, x):
        feat = self.features(x)
        feat = torch.flatten(feat, 1)
        cls_scores = self.classifier(feat)
        bbox_deltas = self.bbox_regressor(feat)
        return cls_scores, bbox_deltas

def train_rcnn(epochs=12, batch_size=32, lr=1e-4):
    print("=================================================================")
    print("      LAB 08 - TASK 2: R-CNN MODEL ARCHITECTURE & TRAINING       ")
    print("      Author: Muhammad John Raza (23k-0069)                     ")
    print("=================================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on compute device: {device}")

    dataset_path = get_path("lab8/outputs/rcnn_proposals_dataset.npz")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Proposals dataset not found at {dataset_path}")

    dataset = RCNNDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    print(f"[*] Loaded {len(dataset)} proposal crops for training across {len(dataloader)} batches/epoch.")

    model = RCNNModel(num_classes=4).to(device)

    cls_criterion = nn.CrossEntropyLoss()
    reg_criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    loss_history = []
    cls_loss_history = []
    reg_loss_history = []
    acc_history = []

    print("\n[*] Starting R-CNN Multi-Task Training Loop:")
    for epoch in range(1, epochs + 1):
        model.train()
        running_total_loss = 0.0
        running_cls_loss = 0.0
        running_reg_loss = 0.0
        correct = 0
        total = 0

        for crops, labels, deltas in dataloader:
            crops = crops.to(device)
            labels = labels.to(device)
            deltas = deltas.to(device)

            optimizer.zero_grad()
            cls_preds, reg_preds = model(crops)

            # Classification loss
            loss_cls = cls_criterion(cls_preds, labels)

            # Regression loss only for foreground positive classes (label > 0)
            fg_mask = (labels > 0)
            if fg_mask.sum() > 0:
                loss_reg = reg_criterion(reg_preds[fg_mask], deltas[fg_mask])
            else:
                loss_reg = torch.tensor(0.0, device=device)

            # Multi-task loss: L_total = L_cls + lambda * L_reg
            loss = loss_cls + 1.5 * loss_reg
            loss.backward()
            optimizer.step()

            running_total_loss += loss.item() * len(labels)
            running_cls_loss += loss_cls.item() * len(labels)
            running_reg_loss += loss_reg.item() * len(labels)

            preds = torch.argmax(cls_preds, dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)

        epoch_loss = running_total_loss / total
        epoch_cls_loss = running_cls_loss / total
        epoch_reg_loss = running_reg_loss / total
        epoch_acc = 100.0 * correct / total

        loss_history.append(epoch_loss)
        cls_loss_history.append(epoch_cls_loss)
        reg_loss_history.append(epoch_reg_loss)
        acc_history.append(epoch_acc)

        print(f"    Epoch [{epoch:02d}/{epochs:02d}] - Loss: {epoch_loss:.4f} (Cls: {epoch_cls_loss:.4f}, Reg: {epoch_reg_loss:.4f}) | Cls Accuracy: {epoch_acc:.2f}%")

    # Save model weights
    model_save_path = os.path.join(output_dir, "rcnn_model.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"\n[*] Saved trained R-CNN model weights to {model_save_path}")

    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    axes[0].plot(range(1, epochs + 1), loss_history, 'b-o', label='Total Multi-Task Loss')
    axes[0].plot(range(1, epochs + 1), cls_loss_history, 'g--', label='Classification Loss')
    axes[0].plot(range(1, epochs + 1), reg_loss_history, 'r:', label='Bounding Box Regression Loss')
    axes[0].set_xlabel('Epochs', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Loss', fontsize=11, fontweight='bold')
    axes[0].set_title('R-CNN Multi-Task Loss vs. Epochs', fontsize=12, fontweight='bold')
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend()

    axes[1].plot(range(1, epochs + 1), acc_history, 'm-s', label='Proposal Classification Accuracy (%)')
    axes[1].set_xlabel('Epochs', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    axes[1].set_title('Proposal Classification Accuracy', fontsize=12, fontweight='bold')
    axes[1].grid(True, linestyle=':', alpha=0.6)
    axes[1].legend()

    plt.suptitle("R-CNN Multi-Task Training Convergence (Classification + BBox Regression)\nAuthor: Muhammad John Raza (23k-0069)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "rcnn_training_loss.png")
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[*] Saved training convergence curves to {plot_path}")

    return model_save_path, plot_path

if __name__ == "__main__":
    train_rcnn()
