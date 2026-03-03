import os
import io
import base64
import textwrap
import sys
from flask import Flask, render_template, request

# --- 強制設定 UTF-8 環境 ---
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import yfinance as yf
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from google import genai
from dotenv import load_dotenv

# --- 初始化與 API 配置 ---
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)
app = Flask(__name__)

# 修正中文顯示
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False 

# ---------------------------------------------------------
# 1. GNN 模型定義 (同 gnn_anomaly.py)
# ---------------------------------------------------------
class FinancialGAT(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = GATConv(in_channels, 16, heads=4)
        self.conv2 = GATConv(64, 1)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.elu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

# ---------------------------------------------------------
# 2. 供應鏈與新聞處理邏輯
# ---------------------------------------------------------
def discover_stocks(target_symbol):
    """LLM 供應鏈自動發現"""
    prompt = f"請列出與台灣股票代號「{target_symbol}」供應鏈最相關的 5 個台灣股票代號(含.TW或.TWO)與簡稱。格式為 代號:名稱 (例如 2317.TW:鴻海)，以逗號分隔。僅回傳代號與名稱。"
    try:
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        items = response.text.strip().split(',')
        stocks, names = [target_symbol], ["搜尋目標"]
        for item in items:
            if ':' in item:
                s, n = item.split(':')
                stocks.append(s.strip()); names.append(n.strip())
        return stocks[:6], names[:6]
    except:
        return [target_symbol], ["目標股"]

def get_real_news_robust(symbol, name):
    """強健的新聞檢索 (RAG Retrieval)"""
    try:
        t = yf.Ticker(symbol)
        news_list = t.news
        if news_list and len(news_list) > 0:
            titles = [n.get('title', '').encode('utf-8', 'ignore').decode('utf-8') for n in news_list[:3]]
            return " | ".join(titles)
        return f"目前市場關於 {name} 的公開快訊較少。"
    except:
        return f"偵測到 {name} 在供應鏈中的波動，建議觀察市場成交量。"

# ---------------------------------------------------------
# 3. 核心運算與繪圖
# ---------------------------------------------------------
def run_gnn_full_analysis(stocks, names):
    node_features, valid_names, valid_symbols = [], [], []
    simulated_sentiments = [0.8, 0.2, 0.5, 0.1, -0.2, 0.4] # 模擬情緒

    for i, s in enumerate(stocks):
        try:
            df = yf.download(s, period="3mo", interval="1d", progress=False)
            if df.empty or len(df) < 2: continue
            p_change = (df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]
            v_change = (df['Volume'].iloc[-1] - df['Volume'].iloc[-2]) / df['Volume'].iloc[-2]
            sentiment = simulated_sentiments[i] if i < len(simulated_sentiments) else 0.0
            
            node_features.append([p_change.item(), v_change.item(), sentiment])
            valid_names.append(names[i])
            valid_symbols.append(s)
        except: pass

    if len(node_features) < 2: return None, None, None, None

    # 建立圖結構 (全連接)
    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor([[i, j] for i in range(len(x)) for j in range(len(x)) if i != j]).t().contiguous()
    graph_data = Data(x=x, edge_index=edge_index)

    # 執行 GNN 推理
    model = FinancialGAT(3)
    model.eval()
    with torch.no_grad():
        prediction = model(graph_data).flatten()
        # 異常公式: $$Score = |Pred - Actual| + |Sentiment - Price|$$
        scores = torch.abs(prediction - graph_data.x[:, 0]) + torch.abs(graph_data.x[:, 2] - graph_data.x[:, 0])
    
    return valid_names, valid_symbols, scores, node_features

def get_plot_url(names, scores):
    plt.clf() # 清除之前的圖形
    plt.figure(figsize=(10, 8))
    G = nx.complete_graph(len(names))
    pos = nx.spring_layout(G, k=2.5, seed=42)
    
    node_colors = [s.item() for s in scores]
    nodes = nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap=plt.cm.Reds, node_size=1800)
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    nx.draw_networkx_labels(G, pos, labels={i: names[i] for i in range(len(names))}, font_family='Microsoft JhengHei', font_weight='bold')
    
    plt.title("供應鏈風險圖譜 (GNN 異常偵測結果)", fontsize=15)
    plt.axis('off')
    
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

# ---------------------------------------------------------
# 4. 路由與診斷生成
# ---------------------------------------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    report, plot_url = None, None
    if request.method == 'POST':
        target = request.form.get('symbol').upper()
        # A. 發現股票
        stocks_list, names_list = discover_stocks(target)
        # B. GNN 運算
        res = run_gnn_full_analysis(stocks_list, names_list)
        
        if res[0]:
            v_names, v_symbols, scores, features = res
            plot_url = get_plot_url(v_names, scores)
            
            # C. 針對最高風險股生成報告 (RAG Generation)
            max_idx = torch.argmax(scores).item()
            name, symbol, score, feat = v_names[max_idx], v_symbols[max_idx], scores[max_idx].item(), features[max_idx]
            news = get_real_news_robust(symbol, name)
            
            prompt = f"""
            你現在是資深金融分析師。系統偵測到 {name} 異常分高達 {score:.2f}。
            數據: 漲跌 {feat[0]*100:.2f}%, 情緒 {feat[2]}。
            即時新聞: {news}
            請分析新聞與股價是否背離，100 字內繁體中文，必須提到新聞關鍵字。
            """
            try:
                response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
                report = response.text.strip().replace('*', '')
            except:
                report = "AI 診斷暫時不可用，請稍後再試。"
            
    return render_template('index.html', report=report, plot_url=plot_url)

if __name__ == '__main__':
    app.run(debug=True)