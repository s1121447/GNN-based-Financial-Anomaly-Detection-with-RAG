import os
import torch
from torch_geometric.data import Data
from graph_builder import discover_supply_chain_graph
from feature_builder import build_feature_store, future_drawdown_label
from config import (
    DATASET_CACHE_DIR,
    FEATURE_COLUMNS,
    LOOKBACK_PERIOD,
    LABEL_HORIZON_DAYS,
    ANOMALY_THRESHOLD,
    MIN_COMMON_DATES
)


def build_single_target_dataset(target_symbol: str):
    graph = discover_supply_chain_graph(target_symbol)
    requested_symbols = [n["symbol"] for n in graph["nodes"]]

    feature_store, raw_store = build_feature_store(requested_symbols, period=LOOKBACK_PERIOD)

    # 依實際可下載 symbol 重建 nodes
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

    if len(real_nodes) < 3:
        print(f"[{target_symbol}] 可用節點不足，略過")
        return []

    # 重建邊
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

    # 找共同日期
    common_dates = None
    for node in real_nodes:
        symbol = node["symbol"]
        idx = set(feature_store[symbol].index)
        common_dates = idx if common_dates is None else common_dates & idx

    common_dates = sorted(common_dates) if common_dates else []
    if len(common_dates) < MIN_COMMON_DATES:
        print(f"[{target_symbol}] 共同日期不足，略過")
        return []

    node_to_idx = {n["symbol"]: i for i, n in enumerate(real_nodes)}

    edge_pairs = []
    for e in real_edges:
        edge_pairs.append([node_to_idx[e["source"]], node_to_idx[e["target"]]])

    if edge_pairs:
        edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    dataset = []

    for dt in common_dates:
        x_rows = []
        y_rows = []

        valid = True
        for node in real_nodes:
            symbol = node["symbol"]
            feat_df = feature_store[symbol]

            if dt not in feat_df.index:
                valid = False
                break

            row = feat_df.loc[dt, FEATURE_COLUMNS]
            if row.isna().any():
                valid = False
                break

            label = future_drawdown_label(
                feat_df["Close"],
                dt,
                horizon_days=LABEL_HORIZON_DAYS,
                threshold=ANOMALY_THRESHOLD
            )
            if label is None:
                valid = False
                break

            x_rows.append(row.values.tolist())
            y_rows.append(label)

        if not valid:
            continue

        data = Data(
            x=torch.tensor(x_rows, dtype=torch.float),
            edge_index=edge_index,
            y=torch.tensor(y_rows, dtype=torch.long)
        )

        data.symbols = [n["symbol"] for n in real_nodes]
        data.names = [n["name"] for n in real_nodes]
        data.roles = [n["role"] for n in real_nodes]
        data.date_str = str(dt.date())
        data.target_symbol = target_symbol

        dataset.append(data)

    print(f"[{target_symbol}] 建立完成，樣本數 = {len(dataset)}")
    return dataset


def build_dataset(target_symbols):
    all_samples = []
    for s in target_symbols:
        all_samples.extend(build_single_target_dataset(s))

    save_path = os.path.join(DATASET_CACHE_DIR, "financial_graph_dataset.pt")
    torch.save(all_samples, save_path)
    print(f"資料集已儲存：{save_path}")
    return save_path


if __name__ == "__main__":
    # 先從少量核心股開始
    targets = ["2330.TW", "2317.TW", "2454.TW"]
    build_dataset(targets)