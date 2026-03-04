import os
import io
import base64
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import yfinance as yf
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify # 增加 jsonify
from google import genai
from dotenv import load_dotenv

load_dotenv()
os.environ["PYTHONUTF8"] = "1"
app = Flask(__name__) 
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False 

class FinancialGAT(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = GATConv(in_channels, 16, heads=4)
        self.conv2 = GATConv(64, 1)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.elu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

def discover_stocks(target_symbol):
    print(f">>>> [1/4] 正在利用 AI 分析供應鏈角色...")
    prompt = f"分析台股「{target_symbol}」並找其供應鏈最相關 10 個股票。格式範例：{target_symbol}:名稱:目標, 2330.TW:台積電:上游, 3680.TWO:健策:下游。僅回傳格式字串。"
    try:
        # 使用 1.5-flash 確保額度穩定
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        text = response.text.strip()
        raw_items = text.replace('\n', ',').split(',')
        stocks, names, roles = [], [], []
        for item in raw_items:
            if item.count(':') == 2:
                s, n, r = item.split(':')
                stocks.append(s.strip().upper()); names.append(n.strip()); roles.append(r.strip())
        return stocks, names, roles
    except Exception as e:
        print(f"AI 分析出錯: {e}")
        return [target_symbol], ["目標股"], ["目標"]

def run_gnn_full_analysis(stocks, names):
    print(f">>>> [2/4] 正在執行數據抓取與精確對齊...")
    price_series_list, success_data = [], []

    for i, s in enumerate(stocks):
        df = None
        current_symbol = s if "." in s else f"{s}.TW"
        for i, s in enumerate(stocks):
            df = None
        # 1. 確保初始格式：如果是純數字則補上 .TW
            current_symbol = s if "." in s else f"{s}.TW"
        
            try:
            # 2. 第一次嘗試下載
                df = yf.download(current_symbol, period="3mo", progress=False)
            
            # 3. 如果沒抓到資料，精確修正後綴
                if (df is None or df.empty or len(df) < 5):
                    if current_symbol.endswith(".TW"):
                    # 如果是 .TW 失敗，換成 .TWO (精確替換最後三個字元)
                        alt_s = current_symbol[:-3] + ".TWO"
                    elif current_symbol.endswith(".TWO"):
                    # 如果是 .TWO 失敗，換成 .TW
                        alt_s = current_symbol[:-4] + ".TW"
                    else:
                        continue
                
                    print(f"嘗試修正代號: {current_symbol} -> {alt_s}")
                    df = yf.download(alt_s, period="3mo", progress=False)
                    if not df.empty and len(df) >= 5:
                        current_symbol = alt_s
            except Exception as e:
                print(f"下載 {current_symbol} 出錯: {e}")
                continue

        # 4. 驗證最終資料是否可用
            if df is None or df.empty or len(df) < 5:
                continue
        try:
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            pct_change = df['Close'].pct_change().dropna()
            if pct_change.empty: continue
            
            pct_series = pct_change.iloc[:, 0] if len(pct_change.shape) > 1 else pct_change
            pct_series.name = names[i]
            price_series_list.append(pct_series)
            
            p_change = (float(df['Close'].iloc[-1]) - float(df['Close'].iloc[-2])) / float(df['Close'].iloc[-2])
            v_change = (float(df['Volume'].iloc[-1]) - float(df['Volume'].iloc[-2])) / float(df['Volume'].iloc[-2]) if float(df['Volume'].iloc[-2]) != 0 else 0
            
            success_data.append({'name': names[i], 'symbol': current_symbol, 'features': [p_change, v_change, 0.5]})
        except: continue

    if len(price_series_list) < 2: return None, None, None, None, None

    combined_df = pd.concat(price_series_list, axis=1, join='outer').fillna(0)
    corr_matrix = combined_df.corr().fillna(0)
    
    final_v_names, final_v_symbols, final_features = [], [], []
    for item in success_data:
        if item['name'] in corr_matrix.columns:
            final_v_names.append(item['name']); final_v_symbols.append(item['symbol']); final_features.append(item['features'])
    
    # --- 核心改動：計算離群分數 ---
    p_changes = torch.tensor([f[0] for f in final_features])
    mean_p = torch.mean(p_changes) # 群體平均漲跌
    # 異常分 = |個體漲跌 - 群體平均| * 放大倍率
    scores = torch.abs(p_changes - mean_p) * 20 
    scores = torch.clamp(scores, max=1.0)
    
    return final_v_names, final_v_symbols, scores, final_features, corr_matrix

def get_plot_url(v_names, v_symbols, scores, v_roles, corr_matrix):
    print(f">>>> [3/4] 正在繪製視覺優化後的風險圖譜...")
    plt.clf(); fig, ax = plt.subplots(figsize=(16, 13)); G = nx.DiGraph()

    for i, name in enumerate(v_names): G.add_node(i, label=f"{name}\n({v_symbols[i]})", role=v_roles[i])
    target_idx = next((i for i, r in enumerate(v_roles) if r == "目標"), None)

    for i in range(len(v_names)):
        for j in range(len(v_names)):
            if i == j: continue
            if (v_roles[i] == "上游" and v_roles[j] == "目標") or (v_roles[i] == "目標" and v_roles[j] == "下游"):
                if v_names[i] in corr_matrix.index and v_names[j] in corr_matrix.columns:
                    w = corr_matrix.loc[v_names[i], v_names[j]]
                    if w > 0.1: G.add_edge(i, j, weight=w)

    isolates = list(nx.isolates(G))
    if target_idx is not None and target_idx in isolates: isolates.remove(target_idx)
    G.remove_nodes_from(isolates)
    
    remaining = sorted(list(G.nodes()))
    u_idx = [i for i in remaining if v_roles[i] == "上游"]
    d_idx = [i for i in remaining if v_roles[i] == "下游"]
    
    pos = {target_idx: (0, 0)} if target_idx is not None else {}
    max_y = 0
    for idx, u in enumerate(u_idx):
        y = ((len(u_idx)-1-idx)-len(u_idx)/2.0+0.5)*2.8; pos[u] = (-1.5, y); max_y = max(max_y, abs(y))
    for idx, d in enumerate(d_idx):
        y = ((len(d_idx)-1-idx)-len(d_idx)/2.0+0.5)*2.8; pos[d] = (1.5, y); max_y = max(max_y, abs(y))

    node_colors = [scores[i].item() for i in remaining]
    nodes = nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap=plt.cm.Reds, node_size=4500, alpha=0.9, vmin=0.0, vmax=1.0)
    
    if G.edges():
        nx.draw_networkx_edges(G, pos, edgelist=G.edges(), width=[max(1.0, G[u][v]['weight']*15) for u,v in G.edges()], alpha=0.3, edge_color='#888', arrows=True, arrowsize=30)
        nx.draw_networkx_edge_labels(G, pos, edge_labels={k: f"{v:.2f}" for k, v in nx.get_edge_attributes(G, 'weight').items()}, font_size=10)

    nx.draw_networkx_labels(G, pos, labels=nx.get_node_attributes(G, 'label'), font_family='Microsoft JhengHei', font_weight='bold', font_size=11)
    
    title_y = max_y + 1.8
    ax.text(-1.5, title_y, "【上游供應】\n(偏差風險)", fontsize=14, ha='center', fontweight='bold')
    ax.text(0.0, title_y, "【核心目標】", fontsize=14, ha='center', fontweight='bold', color='red')
    ax.text(1.5, title_y, "【下游需求】\n(偏差風險)", fontsize=14, ha='center', fontweight='bold')
    
    plt.colorbar(nodes, fraction=0.03, pad=0.04, label='異常離群指數')
    ax.set_ylim(-title_y-1, title_y+1); plt.axis('off')
    
    buf = io.BytesIO(); plt.savefig(buf, format='png', bbox_inches='tight', dpi=100); buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode()

# --- 互動分析路由 ---
@app.route('/analyze_individual/<symbol>')
def analyze_individual(symbol):
    prompt = f"分析台股 {symbol}。請針對該股在今日供應鏈中的表現提供 100 字繁體中文診斷。請評估其今日走勢是漲還是跌。不准用星號。"
    try:
        resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        return jsonify({"report": resp.text.strip().replace('*', '')})
    except Exception as e:
        return jsonify({"report": f"分析失敗: {e}"}), 500

@app.route('/', methods=['GET', 'POST'])
def index():
    report, plot_url, stocks_info = None, None, []
    if request.method == 'POST':
        target = request.form.get('symbol').strip().upper()
        if "." not in target: target += ".TW"
        
        stocks_list, names_list, roles_list = discover_stocks(target)
        res = run_gnn_full_analysis(stocks_list, names_list)
        
        if res and res[0]:
            v_names, v_symbols, scores, features, corr_matrix = res
            
            # --- 核心修正：統一使用「去後綴」的代號進行比對 ---
            v_roles = []
            # 建立一個乾淨的對照表 (全部拿掉 .TW 或 .TWO)
            clean_stocks_list = [s.split('.')[0] for s in stocks_list]
            
            for s in v_symbols:
                clean_s = s.split('.')[0] # 目前處理中的股票
                if clean_s in clean_stocks_list:
                    # 找到在原始名單中的索引位置
                    idx = clean_stocks_list.index(clean_s)
                    v_roles.append(roles_list[idx])
                else:
                    v_roles.append("其他")
            
            # 繪圖
            plot_url = get_plot_url(v_names, v_symbols, scores, v_roles, corr_matrix)
            stocks_info = list(zip(v_symbols, v_names))
            
            # 找出最不合群的人 (離群值最高者)
            max_idx = torch.argmax(scores).item()
            target_name = v_names[max_idx]
            
            # --- AI 診斷生成 ---
            prompt = f"分析對象：{target_name} ({v_symbols[max_idx]})。該股目前與其供應鏈群體表現出現顯著背離。請以金融分析師身份提供 100 字繁體中文診斷，說明此背離現象可能的風險傳導方向。不准使用星號。"
            try:
                # 建議使用 1.5-flash 避免 429 額度錯誤
                resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
                report = resp.text.strip().replace('*', '')
            except Exception as e:
                print(f"AI 報告生成失敗: {e}")
                report = "目前市場波動劇烈，請點擊下方按鈕進行個別深度分析。"
        else:
            report = "數據不足，可能是多檔股票查無資料或代號後綴錯誤。"
            
    return render_template('index.html', report=report, plot_url=plot_url, stocks_info=stocks_info)

if __name__ == '__main__':
    app.run(port=5000, debug=True)