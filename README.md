# 金融供應鏈異常偵測系統

本專案是一套結合 **大型語言模型（LLM）**、**供應鏈圖譜建構**、**市場特徵分析** 與 **機器學習 / 圖神經網路（GNN）** 的台股供應鏈異常偵測系統。

使用者輸入台股代號後，系統會：

1. 透過 LLM 動態建立上下游供應鏈子圖  
2. 擷取各節點公司的市場特徵  
3. 使用模型推論節點異常風險  
4. 以圖譜、排序表與 AI 報告方式呈現結果  

---

## 專案目標

一般投資人多半只看單一股票價格，但實際上風險常沿著供應鏈傳導。  
本作品希望從供應鏈角度出發，建立一套能協助使用者觀察：

- 哪些節點風險偏高
- 風險可能如何傳到核心公司
- 哪些上下游公司值得優先關注

---

## 專案定位

本作品是一套以**供應鏈圖結構為核心**的金融異常偵測原型系統。

目前實驗結果顯示：

- **Random Forest** 是現階段 anomaly F1-score 最佳的模型
- **GNN / GAT** 則作為結合供應鏈圖結構的研究方向
- **LLM** 負責動態建立供應鏈圖與生成可讀性分析報告

因此，本作品的核心不是單押某一個模型，而是建立一套完整的 **AI 金融供應鏈異常偵測系統**，並比較不同模型在此任務上的表現。

---

## 系統特色

- 輸入單一台股代號，自動建立供應鏈子圖
- 使用 Gemini 動態建立上下游關係
- 使用 yfinance 擷取股價與成交量資料
- 比較 Logistic Regression、Random Forest、GAT 模型表現
- 使用 Flask 建立互動式前端
- 自動繪製供應鏈風險圖譜
- 提供節點級 AI 分析報告
- 支援盤後快取推論結果，降低重複計算成本
- 支援節點風險排序表與風險判定顯示

---

## 系統架構

本系統分為三個主要模組：

### 1. LLM 建圖模組
- 使用者輸入股票代號
- Gemini 搜尋並建立目標公司之供應鏈子圖
- 輸出節點（上游 / 目標 / 下游）與邊關係

### 2. 特徵建構與模型模組
- 擷取各節點的歷史市場特徵
- 建立圖結構資料集
- 使用不同模型進行節點異常分類
- 輸出每個節點的異常風險分數

### 3. Flask 視覺化模組
- 顯示供應鏈風險圖譜
- 顯示節點異常程度
- 顯示節點風險排序表
- 顯示 AI 自動化分析報告

---

## 系統流程

```text
使用者輸入股票代號
        ↓
Gemini 動態建構供應鏈圖
        ↓
擷取各節點市場特徵
        ↓
建立圖資料集
        ↓
模型推論（LR / RF / GAT）
        ↓
輸出異常機率、風險排序表、視覺化圖譜與 AI 報告
```

## 特徵設計

目前節點特徵包含：

- 1 日報酬率

- 5 日報酬率

- 20 日報酬率

- 1 日成交量變化率

- 5 日成交量均值比

- 5 日波動率

- 20 日波動率

- MA5 偏離

- MA20 偏離

- RSI(14)

- 高低價區間比

## 標籤定義

目前模型以節點分類方式進行訓練。

若某節點在未來 5 個交易日內最大跌幅小於等於 -8%，則標記為異常節點（label = 1），否則為正常節點（label = 0）。

## 模型說明
### 1. Logistic Regression

Logistic Regression 為基本 baseline 模型，只使用節點本身的表格特徵進行分類，不考慮圖結構資訊。

用途：

- 作為最基本的機器學習對照組

- 驗證單純線性分類器在本任務上的表現

### 2. Random Forest

Random Forest 為目前表現最佳的模型，使用多棵決策樹集成方式進行節點異常分類。

用途：

- 作為目前系統的主要預測模型

- 擅長處理表格型金融特徵

- 能有效捕捉非線性特徵關係

### 3. GAT / GNN

GAT（Graph Attention Network）為圖神經網路模型，除了節點自身特徵外，也利用供應鏈圖中的鄰居節點資訊進行推論。

用途：

- 作為供應鏈圖結構學習的研究方向

- 探討上下游鄰居資訊是否能提升異常偵測能力

- 作為後續圖結構優化的延伸路線

## 實驗結果
| 模型                           |        Accuracy |    異常 Precision |       異常 Recall |           異常 F1 |
| ---------------------------- | --------------: | --------------: | --------------: | --------------: |
| Logistic Regression          |          0.9237 |          0.0000 |          0.0000 |          0.0000 |
| Random Forest                |          0.7474 |          0.1642 |          0.5885 |          0.2567 |
| GAT（未加權）                     |          0.9268 |          1.0000 |          0.0123 |          0.0244 |
| GAT（加權 loss + anomaly-F1 選模） | 0.7892 ~ 0.8633 | 0.1500 ~ 0.2012 | 0.2840 ~ 0.3951 | 0.2174 ~ 0.2355 |

目前實驗結果顯示：

- Logistic Regression 與未加權 GAT 雖然 accuracy 較高，但對異常類別幾乎沒有辨識能力

- 加入 class weights 後，GAT 在 anomaly recall 與 F1-score 上有明顯改善

- 在目前設定下，Random Forest 仍然是 anomaly F1-score 最佳的模型

這表示目前的圖結構設計與特徵設計，尚未完全發揮圖神經網路的優勢。

## Threshold 分析

本研究進一步對驗證集進行 threshold sweep，以將節點異常機率轉換為二元異常預測。

結果顯示目前較佳 threshold 約為 0.50。

這表示在目前模型設定下，0.50 能在 precision 與 recall 間取得較佳平衡；若 threshold 過低，會造成大量誤報；若 threshold 過高，則會漏掉大部分異常節點。

## AI 輕量化設計

本作品目前採用以下兩種輕量化策略：

### 1. 模型輕量化

- 將 GAT 隱藏層維度由 32 縮減為 16

- 在模型參數量下降的情況下，anomaly F1 幾乎維持相近水準

- 顯示本系統可在降低模型複雜度的同時，保留主要異常偵測能力

### 2. 系統輕量化

- 13:30 前採用即時推論模式

- 13:30 後優先讀取當日 cache

- 若無 cache，才重新推論並寫入結果

此設計可降低重複推論與 API 呼叫成本，提升整體系統效率與實務可用性。

### 專案架構
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
├─ train_baseline_rf.py
├─ threshold_search.py
├─ inference.py
├─ requirements.txt
├─ .env
├─ cache/
│  ├─ graphs/
│  ├─ datasets/
│  └─ inference_results/
├─ checkpoints/
│  └─ gnn_model.pt
└─ templates/
   └─ index.html
```
### 使用技術
- Python

- Flask

- Google Gemini API

- yfinance

- PyTorch

- PyTorch Geometric

- NetworkX

- Matplotlib

- Pandas / NumPy

- scikit-learn

## 安裝方式
### 1. 建立環境

建議使用 conda：
```text
conda create -n stock_gnn python=3.11 -y
conda activate stock_gnn
```

### 2.安裝套件
```text
pip install -r requirements.txt
```
### 3. 建立 .env
請在專案根目錄建立 .env：
```text
GEMINI_API_KEY=你的_GEMINI_API_KEY
```
## 執行流程
### Step 1. 建立資料集
```text
python dataset_builder.py
```
這一步會：

- 使用 Gemini 建立目標股票供應鏈子圖

- 下載歷史市場資料

- 建立圖資料集

輸出到：
```text
cache/datasets/financial_graph_dataset.pt
```
## 各模型的用法
### A. Logistic Regression

訓練 Logistic Regression baseline：
```text
python train_baseline.py
```
用途：

- 建立最基本的 baseline

- 比較線性模型在異常偵測任務中的表現

### B. Random Forest

訓練 Random Forest baseline：
```text
python train_baseline_rf.py
```
用途：

- 作為目前主模型的效能比較依據

- 目前 anomaly F1-score 最佳

## C. GAT / GNN

訓練 GAT 模型：
```text
python train_gnn.py
```
用途：

- 研究供應鏈圖結構是否有助於提升異常偵測能力

- 目前為延伸研究方向

訓練完成後會輸出模型到：
```text
checkpoints/gnn_model.pt
```

### D. Threshold 分析

對驗證集進行 threshold sweep：
```text
python threshold_search.py
```

用途：

- 找出較佳 anomaly threshold

- 作為前端風險判定依據

### E. 啟動 Flask 系統
```text
python app.py
```
然後打開瀏覽器：

http://127.0.0.1:5000

## 使用方式

- 在首頁輸入台股代號，例如 2330

- 系統自動建立供應鏈子圖

- 模型對節點進行異常風險推論

- 顯示供應鏈風險圖譜

- 顯示節點風險排序表與風險判定

- 點擊各公司標籤可查看個別 AI 診斷報告

## 系統目前模式

- 13:30 前：盤中即時推論模式

- 13:30 後：盤後快取模式

- 盤後模式下，系統會優先讀取當日推論結果 cache；若 cache 不存在，才重新推論並自動儲存。

## 專案限制

- 供應鏈圖目前由 LLM 動態生成，結果可能受生成品質影響

- 訓練資料目前主要來自少數核心台股，對未納入訓練範圍之股票仍屬探索性推論

- GNN 目前尚未超越 Random Forest baseline

- 新聞特徵實驗曾受 API 配額限制影響，尚未納入穩定主版本

- 部分台股需自動切換 .TW / .TWO 後綴

## 未來工作

- 擴大訓練股票範圍

- 導入更多產業與供應鏈樣本

- 引入 rolling window 時序特徵

- 比較更多模型（如 XGBoost、LightGBM）

- 改善圖結構設計與邊關係定義

- 補強新聞 / 事件風險特徵

- 持續提升 GNN 在 anomaly F1-score 上的表現

## 注意事項

本專案僅供研究、學習、作品展示與競賽使用，
不構成任何投資建議。
