import os
import json
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

NEWS_CACHE_PATH = "cache/news_features.json"


def load_news_cache():
    if os.path.exists(NEWS_CACHE_PATH):
        with open(NEWS_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_news_cache(cache):
    os.makedirs("cache", exist_ok=True)
    with open(NEWS_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def score_news_with_gemini(symbol, company_name, news_text):
    prompt = f"""
你是一個金融風險分析器。請根據以下台股公司新聞內容，輸出 JSON：

{{
  "event_risk": 0到1之間的小數,
  "news_sentiment": -1到1之間的小數
}}

定義：
- event_risk：事件風險強度，越高表示越可能帶來異常波動、供應鏈衝擊、財務風險或市場不確定性
- news_sentiment：新聞情緒，負面為接近 -1，正面為接近 1，中性接近 0

公司：{company_name} ({symbol})

新聞內容：
{news_text}

只回傳 JSON，不要其他說明。
""".strip()

    try:
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        text = resp.text.strip()

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            data = json.loads(text[start:end+1])
            return {
                "event_risk": float(data.get("event_risk", 0.0)),
                "news_sentiment": float(data.get("news_sentiment", 0.0))
            }
    except Exception as e:
        print(f"[News scoring error] {symbol}: {e}")

    return {
        "event_risk": 0.0,
        "news_sentiment": 0.0
    }