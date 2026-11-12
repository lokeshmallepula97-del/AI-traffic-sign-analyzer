"""
Training and Evaluation Pipeline for BanTrafficNet on Official Indian Traffic Sign Dataset.
Executes genuine training, tracks history, evaluates on held-out test split,
and exports model weights and metrics artifacts.
"""

import os
import sys
import json
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from backend.app.models.bantrafficnet_torch import BanTrafficNetTorch, count_parameters

def train_bantrafficnet(epochs: int = 25, batch_size: int = 32, lr: float = 0.001):
    print("=" * 65)
    print("STARTING BANTRAFFICNET TRAINING PIPELINE (INDIAN DATASET)")
    print("=" * 65)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training Device: {device}")
    
    # Paths
    base_dir = Path(__file__).resolve().parent.parent
    train_dir = base_dir / "data" / "train"
    val_dir = base_dir / "data" / "validation"
    test_dir = base_dir / "data" / "test"
    
    artifacts_model_dir = base_dir / "artifacts" / "model"
    artifacts_reports_dir = base_dir / "artifacts" / "reports"
    artifacts_model_dir.mkdir(parents=True, exist_ok=True)
    artifacts_reports_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Transforms
    # Paper specs: 96x96 RGB, normalized [0, 1]
    # Augmentation: rotation (+-15 deg), brightness/contrast (+-10%), translation (+-10%)
    train_transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor() # Automatically scales pixel values to [0.0, 1.0]
    ])
    
    eval_transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor()
    ])
    
    print("[*] Loading datasets...")
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=eval_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=eval_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    num_classes = len(train_dataset.classes)
    print(f"[+] Loaded {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test samples.")
    print(f"[+] Total Detected Classes: {num_classes}")
    
    # 2. Build BanTrafficNet
    model = BanTrafficNetTorch(num_classes=num_classes).to(device)
    total_params = count_parameters(model)
    print(f"[+] BanTrafficNet Parameter Count: {total_params:,}")
    
    # 3. Loss & Optimizer (Adam with lr=0.001 matching paper)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    
    # 4. Training Loop
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "learning_rates": []
    }
    
    best_val_acc = 0.0
    best_model_path = artifacts_model_dir / "bantrafficnet_indian.pth"
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()
            
        train_epoch_loss = running_loss / total_train
        train_epoch_acc = (correct_train / total_train) * 100.0
        
        # Validation
        model.eval()
        val_running_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total_val += labels.size(0)
                correct_val += predicted.eq(labels).sum().item()
                
        val_epoch_loss = val_running_loss / total_val
        val_epoch_acc = (correct_val / total_val) * 100.0
        
        current_lr = optimizer.param_groups[0]['lr']
        history["train_loss"].append(train_epoch_loss)
        history["train_acc"].append(train_epoch_acc)
        history["val_loss"].append(val_epoch_loss)
        history["val_acc"].append(val_epoch_acc)
        history["learning_rates"].append(current_lr)
        
        scheduler.step(val_epoch_acc)
        
        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"Train Loss: {train_epoch_loss:.4f} | Train Acc: {train_epoch_acc:.2f}% | "
              f"Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_epoch_acc:.2f}% (LR: {current_lr:.6f})")
        
        # Checkpoint Best Model
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_epoch_acc,
                'classes': train_dataset.classes,
                'num_classes': num_classes,
                'total_params': total_params
            }, best_model_path)
            print(f"  --> Saved new best model checkpoint (Val Acc: {val_epoch_acc:.2f}%)")
            
    total_duration = time.time() - start_time
    print(f"\n[+] Training complete in {total_duration/60:.2f} minutes. Best Val Acc: {best_val_acc:.2f}%")
    
    # 5. Held-Out Test Split Evaluation
    print("\n[*] Evaluating on Unseen Held-Out Test Set...")
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    y_true = []
    y_pred = []
    
    test_start = time.time()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(predicted.cpu().numpy().tolist())
            
    test_eval_duration = time.time() - test_start
    test_acc = (sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)) * 100.0
    
    # Build Confusion Matrix & Per-Class Stats
    cm = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
        
    per_class_metrics = {}
    precisions = []
    recalls = []
    f1s = []
    
    for i, cls in enumerate(train_dataset.classes):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(num_classes) if r != i)
        fn = sum(cm[i][c] for c in range(num_classes) if c != i)
        
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        
        per_class_metrics[cls] = {
            "precision": round(p * 100, 2),
            "recall": round(r * 100, 2),
            "f1_score": round(f1 * 100, 2),
            "test_samples": sum(cm[i])
        }
        
    macro_precision = (sum(precisions) / len(precisions)) * 100.0
    macro_recall = (sum(recalls) / len(recalls)) * 100.0
    macro_f1 = (sum(f1s) / len(f1s)) * 100.0
    
    # Measure latency
    dummy_input = torch.randn(1, 3, 96, 96).to(device)
    latencies = []
    for _ in range(50):
        t0 = time.perf_counter()
        _ = model(dummy_input)
        latencies.append((time.perf_counter() - t0) * 1000)
    mean_latency_ms = sum(latencies) / len(latencies)
    
    metrics_summary = {
        "model_architecture": "BanTrafficNet",
        "dataset_name": "Indian Traffic Signboards Dataset (56 Classes)",
        "total_params": total_params,
        "input_resolution": "96x96x3",
        "training_epochs": epochs,
        "training_duration_seconds": round(total_duration, 2),
        "best_epoch": checkpoint['epoch'],
        "final_train_acc": round(history["train_acc"][-1], 2),
        "final_train_loss": round(history["train_loss"][-1], 4),
        "best_val_acc": round(best_val_acc, 2),
        "test_acc": round(test_acc, 2),
        "macro_precision": round(macro_precision, 2),
        "macro_recall": round(macro_recall, 2),
        "macro_f1": round(macro_f1, 2),
        "mean_inference_latency_ms": round(mean_latency_ms, 2),
        "model_file_size_mb": round(os.path.getsize(best_model_path) / (1024 * 1024), 2),
        "history": history,
        "confusion_matrix": cm,
        "per_class_metrics": per_class_metrics,
        "classes": train_dataset.classes
    }
    
    metrics_file = artifacts_reports_dir / "evaluation_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics_summary, f, indent=2)
        
    print("\n" + "=" * 65)
    print("FINAL TEST EVALUATION RESULTS:")
    print(f"  Test Accuracy:    {test_acc:.2f}%")
    print(f"  Macro Precision:  {macro_precision:.2f}%")
    print(f"  Macro Recall:     {macro_recall:.2f}%")
    print(f"  Macro F1-Score:   {macro_f1:.2f}%")
    print(f"  Inference Speed:  {mean_latency_ms:.2f} ms ({1000/mean_latency_ms:.1f} FPS)")
    print(f"  Metrics Saved to: {metrics_file}")
    print("=" * 65)
    
    return metrics_summary

if __name__ == "__main__":
    train_bantrafficnet(epochs=15, batch_size=32, lr=0.001)
