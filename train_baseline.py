import os
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from config import DATASET_CACHE_DIR

def flatten_dataset(dataset):
    X, y = [], []
    for data in dataset:
        X.append(data.x.numpy())
        y.append(data.y.numpy())
    X = np.vstack(X)
    y = np.concatenate(y)
    return X, y

def main():
    dataset_path = os.path.join(DATASET_CACHE_DIR, "financial_graph_dataset.pt")
    dataset = torch.load(dataset_path, map_location="cpu", weights_only=False)

    split = int(len(dataset) * 0.8)
    train_set = dataset[:split]
    val_set = dataset[split:]

    X_train, y_train = flatten_dataset(train_set)
    X_val, y_val = flatten_dataset(val_set)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    pred = clf.predict(X_val)
    acc = accuracy_score(y_val, pred)

    print("Baseline Logistic Regression Accuracy:", acc)
    print(classification_report(y_val, pred))

if __name__ == "__main__":
    main()