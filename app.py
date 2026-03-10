import os
import io
import base64
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from google import genai

from inference import run_inference
from config import GEMINI_MODEL

load_dotenv()
app = Flask(__name__)
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]
plt.rcParams["axes.unicode_minus"] = False


def build_plot_url(infer_result):
    raw_symbols = infer_result["raw_symbols"]
    raw_names = infer_result["raw_names"]
    raw_roles = infer_result["raw_roles"]
    edge_meta = infer_result["edge_meta"]

    score_map = {(n["symbol"], n["name"]): n["score"] for n in infer_result["nodes"]}

    G = nx.DiGraph()
    for i, (symbol, name, role) in enumerate(zip(raw_symbols, raw_names, raw_roles)):
        score = score_map[(symbol, name)]
        G.add_node(i, symbol=symbol, name=name, role=role, score=score, label=f"{name}\n({symbol})")

    symbol_to_idx = {s: i for i, s in enumerate(raw_symbols)}
    for e in edge_meta:
        if e["source"] in symbol_to_idx and e["target"] in symbol_to_idx:
            G.add_edge(
                symbol_to_idx[e["source"]],
                symbol_to_idx[e["target"]],
                weight=e["corr"]
            )

    fig, ax = plt.subplots(figsize=(16, 12))

    target_idx = next((i for i, role in enumerate(raw_roles) if role == "目標"), None)
    pos = {}
    upstream = [i for i, role in enumerate(raw_roles) if role == "上游"]
    downstream = [i for i, role in enumerate(raw_roles) if role == "下游"]
    others = [i for i, role in enumerate(raw_roles) if role == "其他"]

    max_y = 0.0
    if target_idx is not None:
        pos[target_idx] = (0.0, 0.0)

    for idx, u in enumerate(upstream):
        y = ((len(upstream)-1-idx) - len(upstream)/2.0 + 0.5) * 2.8
        pos[u] = (-1.8, y)
        max_y = max(max_y, abs(y))

    for idx, d in enumerate(downstream):
        y = ((len(downstream)-1-idx) - len(downstream)/2.0 + 0.5) * 2.8
        pos[d] = (1.8, y)
        max_y = max(max_y, abs(y))

    for idx, o in enumerate(others):
        y = ((len(others)-1-idx) - len(others)/2.0 + 0.5) * 2.4 + 4.0
        pos[o] = (0.0, y)
        max_y = max(max_y, abs(y))

    if max_y < 3:
        max_y = 3

    remaining = list(G.nodes())
    node_colors = [G.nodes[i]["score"] for i in remaining]

    nodes = nx.draw_networkx_nodes(
        G, pos,
        nodelist=remaining,
        node_color=node_colors,
        cmap=plt.cm.Reds,
        node_size=4200,
        alpha=0.92,
        vmin=0.0,
        vmax=1.0,
        ax=ax
    )

    if len(G.edges()) > 0:
        widths = [max(1.0, abs(G[u][v]["weight"]) * 10) for u, v in G.edges()]
        nx.draw_networkx_edges(
            G, pos,
            edgelist=list(G.edges()),
            width=widths,
            alpha=0.35,
            edge_color="#888888",
            arrows=True,
            arrowsize=25,
            ax=ax
        )

        edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges()}
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels=edge_labels,
            font_size=9,
            font_color="#666666",
            ax=ax
        )

    labels = {i: G.nodes[i]["label"] for i in G.nodes()}
    nx.draw_networkx_labels(
        G, pos,
        labels=labels,
        font_family="Microsoft JhengHei",
        font_weight="bold",
        font_size=10,
        ax=ax
    )

    title_y = max_y + 1.8
    ax.text(-1.8, title_y, "【上游供應】\n(異常風險)", ha="center", fontsize=14, fontweight="bold")
    ax.text(0.0, title_y, "【核心目標】", ha="center", fontsize=14, fontweight="bold", color="red")
    ax.text(1.8, title_y, "【下游需求】\n(異常風險)", ha="center", fontsize=14, fontweight="bold")

    cbar = plt.colorbar(nodes, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("GNN 異常機率", rotation=270, labelpad=20)

    ax.set_xlim(-2.6, 2.6)
    ax.set_ylim(-title_y - 1, title_y + 1)
    ax.axis("off")
    plt.title(f"GNN 金融供應鏈異常偵測圖譜（{infer_result['date']}）", fontsize=18, pad=40)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def generate_group_report(infer_result):
    top = infer_result["nodes"][0]
    prompt = f"""
請用繁體中文，以金融分析師口吻，根據以下 GNN 推論結果寫 100~150 字摘要。
不要用星號。

目標股：{infer_result['target_symbol']}
日期：{infer_result['date']}
最高風險節點：{top['name']} ({top['symbol']})
角色：{top['role']}
GNN 異常機率：{top['score']:.3f}

請解釋這代表什麼、可能的供應鏈意義、使用者該注意什麼。
""".strip()

    try:
        resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        return resp.text.strip().replace("*", "")
    except Exception:
        return "GNN 已偵測到高風險節點，但暫時無法生成 AI 摘要。"


@app.route("/analyze_individual/<symbol>")
def analyze_individual(symbol):
    name = request.args.get("name", "該股票")
    prompt = f"""
請用繁體中文，以金融分析師身份分析台股 {name} ({symbol})。
字數 100 字左右，不要用星號。
請從供應鏈位置、近期風險、可能市場解讀三個角度簡短說明。
""".strip()

    try:
        time.sleep(1)
        resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        return jsonify({"report": resp.text.strip().replace("*", "")})
    except Exception:
        return jsonify({"report": "分析暫時無法生成。"}), 500


@app.route("/", methods=["GET", "POST"])
def index():
    report = None
    plot_url = None
    stocks_info = []

    if request.method == "POST":
        target = request.form.get("symbol", "").strip().upper()
        if not target:
            report = "請輸入股票代號。"
            return render_template("index.html", report=report, plot_url=plot_url, stocks_info=stocks_info)

        if "." not in target:
            target += ".TW"

        try:
            infer_result = run_inference(target)
            plot_url = build_plot_url(infer_result)
            report = generate_group_report(infer_result)
            stocks_info = [(n["symbol"], n["name"]) for n in infer_result["nodes"]]
        except Exception as e:
            report = f"推論失敗：{str(e)}"

    return render_template("index.html", report=report, plot_url=plot_url, stocks_info=stocks_info)


if __name__ == "__main__":
    app.run(port=5000, debug=True)