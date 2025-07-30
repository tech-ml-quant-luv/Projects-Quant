import pandas as pd
from envs.trading_env import TradingEnv
from utils.features import add_technical_indicators
from utils.load_data import load_btc_data

df = load_btc_data("data/BTC-USD_data.csv")

env = TradingEnv(df, window_size=50)

obs, _ = env.reset()
print("Observation shape:", obs.shape)
done = False

while not done:
    action = env.action_space.sample()
    obs, reward, done, _, _ = env.step(action)
    env.render()
