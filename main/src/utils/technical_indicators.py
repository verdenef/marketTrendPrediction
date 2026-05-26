"""Technical indicator calculations for OHLCV market data."""

from __future__ import annotations

import pandas as pd

DEFAULT_SMA_WINDOW = 14
DEFAULT_RSI_WINDOW = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9


def add_sma(df: pd.DataFrame, window: int = DEFAULT_SMA_WINDOW) -> pd.DataFrame:
    out = df.copy()
    out[f"SMA_{window}"] = out["Close"].rolling(window=window, min_periods=window).mean()
    return out


def add_rsi(df: pd.DataFrame, window: int = DEFAULT_RSI_WINDOW) -> pd.DataFrame:
    out = df.copy()
    delta = out["Close"].diff()
    gain = delta.clip(lower=0).rolling(window=window, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window=window, min_periods=window).mean()
    rs = gain / loss.replace(0, pd.NA)
    out[f"RSI_{window}"] = 100 - (100 / (1 + rs))
    return out


def add_macd(
    df: pd.DataFrame,
    fast: int = MACD_FAST,
    slow: int = MACD_SLOW,
    signal: int = MACD_SIGNAL,
) -> pd.DataFrame:
    out = df.copy()
    ema_fast = out["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = out["Close"].ewm(span=slow, adjust=False).mean()
    out["MACD"] = ema_fast - ema_slow
    out["MACD_signal"] = out["MACD"].ewm(span=signal, adjust=False).mean()
    out["MACD_hist"] = out["MACD"] - out["MACD_signal"]
    return out


def add_lagged_close(df: pd.DataFrame, lags: tuple[int, ...] = (1,)) -> pd.DataFrame:
    out = df.copy()
    for lag in lags:
        out[f"Close_lag{lag}"] = out["Close"].shift(lag)
    return out


def add_all_indicators(
    df: pd.DataFrame,
    sma_window: int = DEFAULT_SMA_WINDOW,
    rsi_window: int = DEFAULT_RSI_WINDOW,
) -> pd.DataFrame:
    out = add_sma(df, sma_window)
    out = add_rsi(out, rsi_window)
    out = add_macd(out)
    out = add_lagged_close(out, (1,))
    return out
