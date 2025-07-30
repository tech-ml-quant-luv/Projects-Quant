import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from utils.load_data import load_btc_data
from envs.trading_env import TradingEnv

def make_env(df):
    def _init():
        env = TradingEnv(df.copy(), window_size=50)
        return Monitor(env)
    return _init

if __name__ == "__main__":  # ✅ Required for Windows multiprocessing

    df = load_btc_data("data/BTC-USD_data.csv")

    num_envs = 4
    env = SubprocVecEnv([make_env(df) for _ in range(num_envs)])
    eval_env = SubprocVecEnv([make_env(df)])

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/",
        log_path="./logs/",
        eval_freq=5000,
        deterministic=True,
        render=False,
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log="./tensorboard/",
    )

    model.learn(total_timesteps=500_000, callback=eval_callback)
    model.save("ppo_trading_bot_parallel")

