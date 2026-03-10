import os
import torch
from torch_geometric.data import Data

from graph_builder import discover_supply_chain_graph
from feature_builder import build_feature_store, recent_edge_correlation
from model import GATNodeClassifier
from config import CHECKPOINT_DIR, FEATURE_COLUMNS, LOOKBACK_PERIOD


def build_latest_graph_data(target_symbol: str):
    graph = discover_supply_chain_graph(target_symbol)
    requested_symbols = [n["symbol"] for n in graph["nodes"]]

    feature_store, raw_store = build_feature_store(requested_symbols, period=LOOKBACK_PERIOD)

    real_nodes = []
    symbol_map = {}

    for node in graph["nodes"]:
        clean = node["symbol"].split(".")[0].upper()
        matched = None
        for real_symbol in feature_store.keys():
            if real_symbol.split(".")[0].upper() == clean:
                matched = real_symbol
                break
        if matched:
            copied = node.copy()
            copied["symbol"] = matched
            real_nodes.append(copied)
            symbol_map[clean] = matched

    if len(real_nodes) < 2:
        raise RuntimeError("可用節點不足，無法推論")

    real_edges = []
    for e in graph["edges"]:
        src_clean = e["source"].split(".")[0].upper()
        dst_clean = e["target"].split(".")[0].upper()
        if src_clean in symbol_map and dst_clean in symbol_map:
            real_edges.append({
                "source": symbol_map[src_clean],
                "target": symbol_map[dst_clean],
                "relation": e["relation"]
            })

    common_dates = None
    for node in real_nodes:
        symbol = node["symbol"]
        idx = set(feature_store[symbol].index)
        common_dates = idx if common_dates is None else common_dates & idx

    common_dates = sorted(common_dates) if common_dates else []
    if not common_dates:
        raise RuntimeError("沒有共同交易日，無法推論")

    latest_dt = common_dates[-1]

    x_rows = []
    for node in real_nodes:
        symbol = node["symbol"]
        row = feature_store[symbol].loc[latest_dt, FEATURE_COLUMNS]
        x_rows.append(row.values.tolist())

    node_to_idx = {n["symbol"]: i for i, n in enumerate(real_nodes)}
    edge_pairs = []
    edge_meta = []

    for e in real_edges:
        src = e["source"]
        dst = e["target"]
        edge_pairs.append([node_to_idx[src], node_to_idx[dst]])
        edge_meta.append({
            "source": src,
            "target": dst,
            "relation": e["relation"],
            "corr": recent_edge_correlation(raw_store, src, dst, window=60)
        })

    if edge_pairs:
        edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    data = Data(
        x=torch.tensor(x_rows, dtype=torch.float),
        edge_index=edge_index
    )
    data.symbols = [n["symbol"] for n in real_nodes]
    data.names = [n["name"] for n in real_nodes]
    data.roles = [n["role"] for n in real_nodes]
    data.date_str = str(latest_dt.date())
    data.target_symbol = target_symbol

    return data, edge_meta


def load_model():
    ckpt_path = os.path.join(CHECKPOINT_DIR, "gnn_model.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError("找不到 gnn_model.pt，請先訓練模型")

    bundle = torch.load(ckpt_path, map_location="cpu")
    model = GATNodeClassifier(
        in_channels=len(bundle["feature_columns"]),
        hidden_channels=bundle["hidden_channels"],
        num_classes=2
    )
    model.load_state_dict(bundle["model_state_dict"])
    model.eval()
    return model


def run_inference(target_symbol: str):
    model = load_model()
    data, edge_meta = build_latest_graph_data(target_symbol)

    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        probs = torch.softmax(logits, dim=1)[:, 1]

    results = []
    for i, symbol in enumerate(data.symbols):
        results.append({
            "symbol": symbol,
            "name": data.names[i],
            "role": data.roles[i],
            "score": float(probs[i].item())
        })

    results = sorted(results, key=lambda x: x["score"], reverse=True)

    return {
        "date": data.date_str,
        "target_symbol": target_symbol,
        "nodes": results,
        "raw_symbols": data.symbols,
        "raw_names": data.names,
        "raw_roles": data.roles,
        "edge_meta": edge_meta
    }