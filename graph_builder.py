import os
import json
import re
from typing import Dict, List
from dotenv import load_dotenv
from google import genai

from config import GRAPH_CACHE_DIR, GEMINI_MODEL

KB_PATH = "knowledge_base.json"

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def normalize_symbol(symbol: str) -> str:
    symbol = symbol.strip().upper()
    if "." not in symbol:
        return f"{symbol}.TW"
    return symbol


def normalize_role(role_text: str) -> str:
    role_text = (role_text or "").strip()
    if "目標" in role_text or "核心" in role_text:
        return "目標"
    if "上游" in role_text or "供應" in role_text or "材料" in role_text or "設備" in role_text:
        return "上游"
    if "下游" in role_text or "需求" in role_text or "客戶" in role_text or "應用" in role_text:
        return "下游"
    return "其他"


def _extract_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise ValueError("Gemini 未回傳可解析 JSON")
        return json.loads(match.group(0))


def build_edges_from_nodes(nodes):
    edges = []
    target_nodes = [n for n in nodes if n["role"] == "目標"]
    if not target_nodes:
        return edges

    target_symbol = target_nodes[0]["symbol"]

    for node in nodes:
        if node["symbol"] == target_symbol:
            continue

        if node["role"] == "上游":
            edges.append({
                "source": node["symbol"],
                "target": target_symbol,
                "relation": "上游供應"
            })
        elif node["role"] == "下游":
            edges.append({
                "source": target_symbol,
                "target": node["symbol"],
                "relation": "下游需求"
            })

    return edges


def discover_supply_chain_graph(target_symbol: str, force_refresh: bool = False) -> dict:
    target_symbol = normalize_symbol(target_symbol)
    target_clean = target_symbol.split(".")[0]
    cache_path = os.path.join(GRAPH_CACHE_DIR, f"{target_clean}.json")

    if os.path.exists(cache_path) and not force_refresh:
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    prompt = f"""
你是一個台股供應鏈圖譜建構器。

請分析台股 {target_symbol} 的供應鏈，找出最相關的上市櫃公司。
請只回傳 JSON，不要有其他說明。

格式如下：
{{
  "target": {{"symbol": "{target_symbol}", "name": "公司名稱", "role": "目標"}},
  "related": [
    {{"symbol": "3680.TWO", "name": "家登", "role": "上游"}},
    {{"symbol": "2454.TW", "name": "聯發科", "role": "下游"}}
  ]
}}

規則：
1. 只能放台灣上市櫃股票，symbol 必須是 .TW 或 .TWO
2. role 只能是 目標、上游、下游
3. related 盡量控制在 8~12 家
4. 目標公司必須是 {target_symbol}
5. 不要加入外國股票代號
""".strip()

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )
    data = _extract_json(response.text)

    target = data.get("target", {})
    related = data.get("related", [])

    nodes = []
    nodes.append({
        "symbol": normalize_symbol(target.get("symbol", target_symbol)),
        "name": target.get("name", target_clean),
        "role": "目標"
    })

    seen = {nodes[0]["symbol"].split(".")[0].upper()}

    for item in related:
        symbol = normalize_symbol(item.get("symbol", ""))
        if not symbol:
            continue

        clean = symbol.split(".")[0].upper()
        if clean in seen:
            continue

        role = normalize_role(item.get("role", "其他"))
        if role == "其他":
            continue

        nodes.append({
            "symbol": symbol,
            "name": item.get("name", clean),
            "role": role
        })
        seen.add(clean)

    edges = build_edges_from_nodes(nodes)

    graph = {
        "target_symbol": target_symbol,
        "nodes": nodes,
        "edges": edges
    }

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)

    return graph