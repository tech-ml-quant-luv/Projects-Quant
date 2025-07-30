import pandas as pd
import numpy as np

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Normalize prices
    df["price_change"] = df["Close"].pct_change().fillna(0)
    df["volatility"] = df["Close"].rolling(10).std().fillna(0)
    df["range"] = (df["High"] - df["Low"]) / df["Open"]

    # RSI
    delta = df["Close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(14).mean()
    avg_loss = pd.Series(loss).rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))

    # EMA
    df["ema_fast"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["ema_slow"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["macd"] = df["ema_fast"] - df["ema_slow"]

    # Drop any initial NaNs
    df = df.dropna().reset_index(drop=True)

    return df
