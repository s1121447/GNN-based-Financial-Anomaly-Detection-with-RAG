from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
import yfinance as yf


def download_stock_df(symbol: str, period: str = "2y") -> Tuple[Optional[str], Optional[pd.DataFrame]]:
    current_symbol = symbol if "." in symbol else f"{symbol}.TW"
    df = yf.download(current_symbol, period=period, progress=False, auto_adjust=False)

    if df is None or df.empty or len(df) < 40:
        if current_symbol.endswith(".TW"):
            alt_symbol = current_symbol[:-3] + ".TWO"
        elif current_symbol.endswith(".TWO"):
            alt_symbol = current_symbol[:-4] + ".TW"
        else:
            alt_symbol = current_symbol

        if alt_symbol != current_symbol:
            alt_df = yf.download(alt_symbol, period=period, progress=False, auto_adjust=False)
            if alt_df is not None and not alt_df.empty and len(alt_df) >= 40:
                current_symbol = alt_symbol
                df = alt_df

    if df is None or df.empty or len(df) < 40:
        return None, None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return current_symbol, df


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / (loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].astype(float)
    volume = df["Volume"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)

    ret = close.pct_change()

    feat = pd.DataFrame(index=df.index)
    feat["ret_1d"] = ret
    feat["ret_5d"] = close.pct_change(5)
    feat["ret_20d"] = close.pct_change(20)
    feat["vol_change_1d"] = volume.pct_change().replace([np.inf, -np.inf], 0)
    feat["vol_ratio_5"] = volume / volume.rolling(5).mean() - 1
    feat["volatility_5"] = ret.rolling(5).std()
    feat["volatility_20"] = ret.rolling(20).std()
    feat["ma5_bias"] = close / close.rolling(5).mean() - 1
    feat["ma20_bias"] = close / close.rolling(20).mean() - 1
    feat["rsi_14"] = compute_rsi(close, 14) / 100.0
    feat["hl_range"] = (high - low) / close.replace(0, np.nan)

    feat["Close"] = close
    feat = feat.replace([np.inf, -np.inf], np.nan).dropna()
    return feat


def build_feature_store(symbols, period="2y") -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    feature_store = {}
    raw_store = {}

    for symbol in symbols:
        final_symbol, df = download_stock_df(symbol, period=period)
        if df is None:
            continue

        feat_df = build_feature_frame(df)
        feature_store[final_symbol] = feat_df
        raw_store[final_symbol] = df.copy()

    return feature_store, raw_store


def future_drawdown_label(close_series: pd.Series, current_date, horizon_days: int = 5, threshold: float = -0.08):
    if current_date not in close_series.index:
        return None

    loc = close_series.index.get_loc(current_date)
    if isinstance(loc, slice):
        return None

    if loc + horizon_days >= len(close_series):
        return None

    current_close = float(close_series.iloc[loc])
    future_window = close_series.iloc[loc + 1: loc + 1 + horizon_days]
    if len(future_window) < horizon_days:
        return None

    min_future = float(future_window.min())
    drawdown = min_future / current_close - 1.0
    return 1 if drawdown <= threshold else 0


def recent_edge_correlation(raw_store: Dict[str, pd.DataFrame], src: str, dst: str, window: int = 60) -> float:
    if src not in raw_store or dst not in raw_store:
        return 0.0

    a = raw_store[src]["Close"].pct_change().dropna()
    b = raw_store[dst]["Close"].pct_change().dropna()

    joined = pd.concat([a, b], axis=1, join="inner").dropna()
    if len(joined) < 20:
        return 0.0

    joined = joined.tail(window)
    return float(joined.corr().iloc[0, 1])