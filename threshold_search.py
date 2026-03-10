import os
import random
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

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


def collect_val_probs(model, loader, device):
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            probs = torch.softmax(out, dim=1)[:, 1]

            all_probs.extend(probs.cpu().numpy().tolist())
            all_labels.extend(batch.y.cpu().numpy().tolist())

    return all_probs, all_labels


def main():
    dataset_path = os.path.join(DATASET_CACHE_DIR, "financial_graph_dataset.pt")
    dataset = torch.load(dataset_path, map_location="cpu", weights_only=False)

    train_set, val_set = split_dataset(dataset)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False)

    ckpt_path = os.path.join(CHECKPOINT_DIR, "gnn_model.pt")
    bundle = torch.load(ckpt_path, map_location="cpu")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GATNodeClassifier(
        in_channels=len(FEATURE_COLUMNS),
        hidden_channels=bundle["hidden_channels"],
        num_classes=2
    ).to(device)

    model.load_state_dict(bundle["model_state_dict"])

    probs, labels = collect_val_probs(model, val_loader, device)

    best_threshold = 0.5
    best_f1 = 0.0

    print("threshold | acc    | precision | recall | f1")
    print("-" * 50)

    for t in [i / 100 for i in range(10, 91, 5)]:
        preds = [1 if p >= t else 0 for p in probs]

        acc = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, labels=[1], average=None, zero_division=0
        )

        p1 = precision[0]
        r1 = recall[0]
        f1_1 = f1[0]

        print(f"{t:8.2f} | {acc:.4f} | {p1:.4f}    | {r1:.4f} | {f1_1:.4f}")

        if f1_1 > best_f1:
            best_f1 = f1_1
            best_threshold = t

    print("\n=== Best Threshold ===")
    print(f"Best threshold: {best_threshold:.2f}")
    print(f"Best anomaly F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()