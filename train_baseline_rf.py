import os
import random
import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

from config import DATASET_CACHE_DIR


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


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


def flatten_dataset(dataset):
    X, y = [], []

    for data in dataset:
        X.append(data.x.numpy())
        y.append(data.y.numpy())

    X = np.vstack(X)
    y = np.concatenate(y)
    return X, y


def main():
    set_seed(42)

    dataset_path = os.path.join(DATASET_CACHE_DIR, "financial_graph_dataset.pt")
    dataset = torch.load(dataset_path, map_location="cpu", weights_only=False)

    if len(dataset) == 0:
        raise RuntimeError("資料集為空，請先執行 dataset_builder.py")

    train_set, val_set = split_dataset(dataset)

    X_train, y_train = flatten_dataset(train_set)
    X_val, y_val = flatten_dataset(val_set)

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    clf.fit(X_train, y_train)
    pred = clf.predict(X_val)

    acc = accuracy_score(y_val, pred)
    report = classification_report(y_val, pred, digits=4, zero_division=0)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_val,
        pred,
        labels=[1],
        average=None,
        zero_division=0
    )

    anomaly_precision = precision[0]
    anomaly_recall = recall[0]
    anomaly_f1 = f1[0]

    print("=== Random Forest Baseline Result ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Anomaly Precision: {anomaly_precision:.4f}")
    print(f"Anomaly Recall: {anomaly_recall:.4f}")
    print(f"Anomaly F1: {anomaly_f1:.4f}")
    print("\nClassification Report:")
    print(report)


if __name__ == "__main__":
    main()