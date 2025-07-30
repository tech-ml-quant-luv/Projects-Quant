import pandas as pd
from utils.features import add_technical_indicators

def load_btc_data(filepath: str):
    df = pd.read_csv(filepath)
    df.columns = ["Datetime", "Open", "High", "Low", "Close", "Volume"]
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.dropna()
    df = df.sort_values("Datetime").reset_index(drop=True)
    df = add_technical_indicators(df)
    return df