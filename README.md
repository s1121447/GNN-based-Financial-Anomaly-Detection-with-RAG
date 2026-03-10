# GNN 金融供應鏈異常偵測系統

結合 **大型語言模型（LLM）**、**供應鏈圖譜建構** 與 **圖神經網路（GNN）** 的台股異常偵測系統。  
本系統讓使用者輸入台股代號後，自動建立該公司上下游供應鏈子圖，擷取各節點的市場特徵，並透過 GNN 模型推論供應鏈節點的異常風險，最後再由 LLM 生成可讀性的金融分析報告。

---

## 專案特色

- 輸入單一台股代號，自動建立供應鏈子圖
- 使用 Gemini 建立上下游節點與供應鏈關係
- 使用 yfinance 擷取歷史股價與成交量資料
- 使用 PyTorch Geometric 建立 GAT 模型進行節點異常分類
- 使用 Flask 建立互動式前端頁面
- 自動繪製供應鏈風險圖譜
- 提供節點級 AI 分析報告

---

## 系統架構

本系統分為三個主要模組：

### 1. LLM 建圖模組
- 使用者輸入股票代號
- Gemini 搜尋並建立目標公司之供應鏈子圖
- 輸出節點（上游 / 目標 / 下游）與邊關係

### 2. GNN 訓練 / 推論模組
- 擷取各節點的歷史市場特徵
- 建立圖結構資料集
- 使用 GAT 模型進行節點異常分類
- 輸出每個節點的異常風險分數

### 3. Flask 視覺化模組
- 顯示供應鏈風險圖譜
- 顯示節點異常程度
- 顯示 AI 自動化分析報告

## 實驗結果

| 模型 | Accuracy | 異常 Precision | 異常 Recall | 異常 F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.9237 | 0.0000 | 0.0000 | 0.0000 |
| Random Forest | 0.7474 | 0.1642 | 0.5885 | 0.2567 |
| GAT（未加權） | 0.9268 | 1.0000 | 0.0123 | 0.0244 |
| GAT（加權 loss + anomaly-F1 選模） | 0.8035 ~ 0.8633 | 0.1561 ~ 0.2012 | 0.2840 ~ 0.3745 | 0.2203 ~ 0.2355 |

目前實驗結果顯示，加權後的 GAT 雖然相較於 Logistic Regression 有明顯改善，但在異常類別 F1-score 上仍未超越 Random Forest。這表示目前的圖結構設計與特徵設計，尚未完全發揮圖神經網路的優勢。


結果顯示，雖然 baseline 與未加權 GAT 在 accuracy 上表現較高，但對異常類別幾乎沒有辨識能力。加入 class weights 後，GAT 在 anomaly recall 與 F1-score 上明顯提升，顯示在高度類別不平衡的金融異常偵測任務中，單看 accuracy 並不足以反映模型的真實效能。

## Threshold 分析

本研究進一步對驗證集進行 threshold sweep，以將節點異常機率轉換為二元異常預測。

結果顯示目前最佳 threshold 為 **0.50**，其異常類別表現為：

- Precision：0.2012
- Recall：0.2840
- F1-score：0.2355

這表示在目前模型設定下，0.50 為 precision 與 recall 的最佳平衡點；若 threshold 過低，會造成大量誤報；若 threshold 過高，則會漏掉大部分異常節點。

---

## 專案架構

```text
financial_gnn/
├─ app.py
├─ config.py
├─ graph_builder.py
├─ feature_builder.py
├─ dataset_builder.py
├─ model.py
├─ train_gnn.py
├─ train_baseline.py
├─ inference.py
├─ requirements.txt
├─ .env
├─ cache/
│  ├─ graphs/
│  └─ datasets/
├─ checkpoints/
│  └─ gnn_model.pt
└─ templates/
   └─ index.html
```
##使用技術<br>
Python<br>
Flask<br>
Google Gemini API<br>
yfinance<br>
PyTorch<br>
PyTorch Geometric<br>
NetworkX<br>
Matplotlib<br>
Pandas / NumPy<br>

特徵設計<br>
目前節點特徵包含：<br>
1 日報酬率<br>
5 日報酬率<br>
20 日報酬率<br>
1 日成交量變化率<br>
5 日成交量均值比<br>
5 日波動率<br>
20 日波動率<br>
MA5 偏離<br>
MA20 偏離<br>
RSI(14)<br>
高低價區間比<br>

標籤定義<br>
目前模型以節點分類方式進行訓練。<br>
若某節點在未來 5 個交易日內最大跌幅小於等於 -8%，則標記為異常節點（label = 1），否則為正常節點（label = 0）。

安裝方式

1. 建立環境

建議使用 conda：
```text
conda create -n stock_gnn python=3.11 -y
conda activate stock_gnn
```
2. 安裝套件
```text
pip install -r requirements.txt
```
3. 建立 .env

請在專案根目錄建立 .env：
```text
GEMINI_API_KEY=你的_GEMINI_API_KEY
```
執行流程

Step 1. 建立資料集
```text
python dataset_builder.py
```
這一步會：

使用 Gemini 建立目標股票供應鏈子圖

下載歷史市場資料

建立圖資料集

輸出到：

cache/datasets/financial_graph_dataset.pt

Step 2. 訓練 GNN 模型
```text
python train_gnn.py
```
訓練完成後會輸出模型到：

checkpoints/gnn_model.pt

Step 3. 啟動 Flask 系統
```text
python app.py
```
然後打開瀏覽器：

http://127.0.0.1:5000
使用方式

在首頁輸入台股代號，例如 2330

系統自動建立供應鏈子圖

GNN 對節點進行異常風險推論

顯示供應鏈風險圖譜

點擊各公司標籤可查看個別 AI 診斷報告
