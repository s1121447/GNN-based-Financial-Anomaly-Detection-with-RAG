📈基於圖神經網路與檢索增強生成之金融異常監測系統

📖專案簡介
本專案開發於 2026 年，旨在解決金融市場中「資訊不對稱」與「輿情言行背離」之風險。系統結合了 圖神經網路 (Graph Neural Networks, GNN) 的空間推理能力與 大型語言模型 (LLM) 的語意理解力，建立一套自動化的供應鏈風險監測平台。

透過 RAG (Retrieval-Augmented Generation) 技術，系統能動態檢索即時財經新聞，並針對 GNN 偵測出的異常節點提供可解釋性的 AI 診斷報告。

🚀核心技術亮點
動態供應鏈發現 (Dynamic RAG Retrieval)：
    利用 Gemini 2.5 Flash 進行語意檢索，輸入單一股票代號即可自動識別其上下游供應鏈節點。
    具備自動後綴校正機制，支援台灣上市 (.TW) 與上櫃 (.TWO) 股票數據自動匹配。
GNN 異常偵測模型 (Spatial Reasoning)：
    採用 GAT (Graph Attention Network) 卷積層，捕捉供應鏈節點間的價格與成交量連動特徵。
    自研異常評分機制：
    $$\text{Anomaly Score} = |f_{\text{GAT}}(X) - \text{Price}| + |\text{Sentiment} - \text{Price}|$$
端對端 Web 交互介面：
    基於 Flask 框架構建，實現搜尋、運算、視覺化與報告生成的一鍵式體驗。
    整合 Matplotlib 高性能渲染技術，動態生成風險熱圖 (Risk Heatmap)。

🛠️技術棧 (Tech Stack)
Deep Learning: PyTorch, PyTorch Geometric (GNN 核心)
LLM Interface: Google GenAI SDK (Gemini 2.0 Flash)
Backend: Flask (Web Server)
Data API: Yahoo Finance (yfinance)
Visualizations: NetworkX, Matplotlib

📦安裝與環境設定
1. 複製專案
    git clone https://github.com/你的帳號/GNN-Stock-Anomaly-Detection.git
    cd GNN-Stock-Anomaly-Detection
2.安裝套件
    pip install -r requirements.txt
3.設定環境變數
    請在根目錄建立 .env 檔案並填入您的 API Key：
        GEMINI_API_KEY=您的_Gemini_API_Key
🖥️ 執行與展示
    python app.py

開啟瀏覽器前往 http://127.0.0.1:5000，輸入台股代號（如 2330.TW）即可開始診斷。


未來展望 (Future Work)
預計將此系統擴充至 多層級供應鏈回溯分析，並嘗試串接真實情緒分析 API 以取代模擬數據。