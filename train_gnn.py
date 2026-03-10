import os
import random
import torch
from torch_geometric.loader import DataLoader

from model import GATNodeClassifier
from config import DATASET_CACHE_DIR, CHECKPOINT_DIR, FEATURE_COLUMNS


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


def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            loss = criterion(out, batch.y)
            pred = out.argmax(dim=1)

            total_loss += loss.item()
            correct += (pred == batch.y).sum().item()
            total += batch.y.numel()

    acc = correct / total if total > 0 else 0.0
    loss = total_loss / len(loader) if len(loader) > 0 else 0.0
    return loss, acc


def main():
    dataset_path = os.path.join(DATASET_CACHE_DIR, "financial_graph_dataset.pt")
    dataset = torch.load(dataset_path, map_location="cpu", weights_only=False)

    if len(dataset) == 0:
        raise RuntimeError("資料集為空，請先執行 dataset_builder.py")

    train_set, val_set = split_dataset(dataset)

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GATNodeClassifier(
        in_channels=len(FEATURE_COLUMNS),
        hidden_channels=32,
        num_classes=2
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0.0
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
        val_loss, val_acc = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "feature_columns": FEATURE_COLUMNS,
                "hidden_channels": 32
            }, best_path)

    print(f"最佳模型已儲存：{best_path}, best_val_acc={best_val_acc:.4f}")


if __name__ == "__main__":
    main()