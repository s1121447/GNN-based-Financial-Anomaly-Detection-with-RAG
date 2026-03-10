import os
import random
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

from model import GATNodeClassifier
from config import DATASET_CACHE_DIR, CHECKPOINT_DIR, FEATURE_COLUMNS
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def split_dataset(dataset, train_ratio=0.8, seed=42):
    random.seed(seed)
    idx = list(range(len(dataset)))
    random.shuffle(idx)

    n_train = int(len(idx) * train_ratio)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:]

    train_set = [dataset[i] for i in train_idx]
    val_set = [dataset[i] for i in val_idx]
    return train_set, val_set


def evaluate(model, loader, device, criterion, threshold=0.5):
    model.eval()
    total_loss = 0.0

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            loss = criterion(out, batch.y)

            probs = torch.softmax(out, dim=1)[:, 1]
            pred = (probs >= threshold).long()

            total_loss += loss.item()
            all_probs.extend(probs.cpu().numpy().tolist())
            all_preds.extend(pred.cpu().numpy().tolist())
            all_labels.extend(batch.y.cpu().numpy().tolist())

    loss = total_loss / len(loader) if len(loader) > 0 else 0.0
    acc = accuracy_score(all_labels, all_preds)

    report = classification_report(
        all_labels,
        all_preds,
        digits=4,
        zero_division=0
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        labels=[1],
        average=None,
        zero_division=0
    )

    anomaly_precision = precision[0]
    anomaly_recall = recall[0]
    anomaly_f1 = f1[0]

    return loss, acc, report, anomaly_precision, anomaly_recall, anomaly_f1


def main():
    set_seed(42)
    dataset_path = os.path.join(DATASET_CACHE_DIR, "financial_graph_dataset.pt")
    dataset = torch.load(dataset_path, map_location="cpu", weights_only=False)

    if len(dataset) == 0:
        raise RuntimeError("資料集為空，請先執行 dataset_builder.py")

    train_set, val_set = split_dataset(dataset)

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 計算 class weights
    train_labels = []
    for data in train_set:
        train_labels.extend(data.y.tolist())

    num_class_0 = sum(1 for y in train_labels if y == 0)
    num_class_1 = sum(1 for y in train_labels if y == 1)

    total = num_class_0 + num_class_1
    weight_0 = total / (2 * num_class_0)
    weight_1 = total / (2 * num_class_1)

    class_weights = torch.tensor([weight_0, weight_1], dtype=torch.float).to(device)
    print("Class weights:", class_weights)

    model = GATNodeClassifier(
        in_channels=len(FEATURE_COLUMNS),
        hidden_channels=32,
        num_classes=2
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    best_anomaly_f1 = 0.0
    best_val_acc = 0.0
    best_val_report = ""
    best_path = os.path.join(CHECKPOINT_DIR, "gnn_model.pt")

    for epoch in range(1, 101):
        model.train()
        total_train_loss = 0.0

        for batch in train_loader:
            batch = batch.to(device)

            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        train_loss = total_train_loss / len(train_loader)
        val_loss, val_acc, val_report, anomaly_precision, anomaly_recall, anomaly_f1 = evaluate(
            model, val_loader, device, criterion, threshold=0.5
        )

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Anomaly F1: {anomaly_f1:.4f}"
        )

        if anomaly_f1 > best_anomaly_f1:
            best_anomaly_f1 = anomaly_f1
            best_val_acc = val_acc
            best_val_report = val_report

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "feature_columns": FEATURE_COLUMNS,
                    "hidden_channels": 32
                },
                best_path
            )

    print("\n=== Best Validation Result ===")
    print(f"Best Anomaly F1: {best_anomaly_f1:.4f}")
    print(f"Best Val Acc: {best_val_acc:.4f}")
    print("Best Validation Classification Report:")
    print(best_val_report)
    print(f"最佳模型已儲存：{best_path}")


if __name__ == "__main__":
    main()