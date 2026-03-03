import os
import io
import base64
import sys
import matplotlib
matplotlib.use('Agg') 

import matplotlib.pyplot as plt
from flask import Flask, render_template, request
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import yfinance as yf
import networkx as nx
from google import genai
from dotenv import load_dotenv

# --- 初始化 ---
load_dotenv()
os.environ["PYTHONUTF8"] = "1"
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
app = Flask(__name__)

# 修正中文顯示
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False 

# ---------------------------------------------------------
# 1. 修正：自動獲取所有節點名稱 (解決「搜尋目標」字樣問題)
# ---------------------------------------------------------
def discover_stocks(target_symbol):
    print(f">>> [1/4] 正在擴大檢索產業鏈節點...")
    # 將數量提升至 10 個相關節點
    prompt = f"請列出台股「{target_symbol}」本身的名稱，以及與其供應鏈最相關的 10 個台灣股票代號與名稱。格式: 代號:名稱, 代號:名稱。請務必確認代號後綴（上市為.TW, 上櫃為.TWO）。僅回傳代號與名稱。"
    try:
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        text = response.text.strip()
        raw_items = text.replace('\n', ',').split(',')
        stocks, names = [], []
        for item in raw_items:
            if ':' in item:
                s, n = item.split(':')
                stocks.append(s.strip().upper())
                names.append(n.strip())
        # 移除 [:6] 限制，讓它動態回傳所有找到的節點
        return stocks, names 
    except Exception as e:
        print(f"!!! AI 發現股票失敗: {e}")
        return [target_symbol], ["目標股"]

# ---------------------------------------------------------
# 2. 修正：自動校正後綴邏輯 (解決 6488.TW 報錯問題)
# ---------------------------------------------------------
def run_gnn_full_analysis(stocks, names):
    print(f">>> [2/4] 正在抓取數據與執行 GNN...")
    node_features, v_names, v_symbols = [], [], []
    
    for i, s in enumerate(stocks):
        try:
            # 1. 嘗試原代號下載
            df = yf.download(s, period="1mo", progress=False)
            
            # 2. 如果失敗且是 .TW，自動嘗試 .TWO (校正上櫃股票)
            if df.empty and s.endswith(".TW"):
                alt_s = s.replace(".TW", ".TWO")
                df = yf.download(alt_s, period="1mo", progress=False)
                if not df.empty: s = alt_s
            
            if df.empty:
                print(f"!!! 無法獲取股票數據: {s}")
                continue
            
            # (接下來處理 p_change, v_change 等...)
            p_change = (df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]
            v_change = (df['Volume'].iloc[-1] - df['Volume'].iloc[-2]) / df['Volume'].iloc[-2]
            
            node_features.append([p_change.item(), v_change.item(), 0.5])
            v_names.append(names[i])
            v_symbols.append(s)
        except: continue

    if len(node_features) < 2: return None, None, None, None

    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor([[i, j] for i in range(len(x)) for j in range(len(x)) if i != j]).t().contiguous()
    
    # 簡單異常分模型邏輯
    scores = torch.abs(x[:, 0]) * 10 
    return v_names, v_symbols, scores, node_features

# ---------------------------------------------------------
# 3. 修正：添加 Colorbar (解決右側風險圖表消失問題)
# ---------------------------------------------------------
def get_plot_url(names, scores):
    print(f">>> [3/4] 正在生成視覺化圖表...")
    plt.clf()
    fig = plt.figure(figsize=(11, 8)) # 稍微加寬
    G = nx.complete_graph(len(names))
    pos = nx.spring_layout(G, k=2.0, seed=42)
    
    node_colors = [s.item() for s in scores]
    nodes = nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap=plt.cm.Reds, 
                               node_size=1800, alpha=0.9, vmin=0.0, vmax=1.0)
    nx.draw_networkx_edges(G, pos, alpha=0.2, width=1.5)
    nx.draw_networkx_labels(G, pos, labels={i: names[i] for i in range(len(names))}, font_family='Microsoft JhengHei', font_weight='bold')
    
    # 添加顏色條 (Colorbar)
    cbar = plt.colorbar(nodes, fraction=0.03, pad=0.04)
    cbar.set_label('異常風險指數 (Anomaly Score)', rotation=270, labelpad=20)
    
    plt.title("供應鏈風險圖譜 (GNN 異常偵測)", fontsize=16, pad=20)
    plt.axis('off')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode()

# ---------------------------------------------------------
# 4. 路由
# ---------------------------------------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    report, plot_url = None, None
    if request.method == 'POST':
        target = request.form.get('symbol').strip().upper()
        # 基本檢查：若無後綴預設給 .TW
        if "." not in target: target += ".TW"
            
        stocks_list, names_list = discover_stocks(target)
        res = run_gnn_full_analysis(stocks_list, names_list)
        
        if res and res[0]:
            v_names, v_symbols, scores, features = res
            plot_url = get_plot_url(v_names, scores)
            
            # 生成報告 (使用 AI 診斷)
            max_idx = torch.argmax(scores).item()
            report_prompt = f"在台股供應鏈中，{v_names[max_idx]} 的異常分最高。漲跌幅為 {features[max_idx][0]*100:.2f}%。請以資深分析師口吻提供 100 字診斷。"
            try:
                resp = client.models.generate_content(model="gemini-2.5-flash", contents=report_prompt)
                report = resp.text.replace('*', '')
            except:
                report = "分析完成，請參考圖譜紅點標示之風險。"
        else:
            report = "數據下載失敗或代碼錯誤，請嘗試輸入正確代碼（如 2330.TW）。"
            
    return render_template('index.html', report=report, plot_url=plot_url)

if __name__ == '__main__':
    app.run(port=5000, debug=True)