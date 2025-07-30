import gymnasium as gym
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, df, window_size=50, initial_balance=10000, feature_cols=None):
        super(TradingEnv, self).__init__()

        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = initial_balance

        # Select features
        self.feature_cols = feature_cols or [
            "price_change", "volatility", "range", "rsi", "macd"
        ]

        self.action_space = gym.spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size, len(self.feature_cols)),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.total_profit = 0
        self.trades = []

        return self._get_observation(), {}

    def _get_observation(self):
        obs = self.df.loc[self.current_step - self.window_size:self.current_step - 1, self.feature_cols]
        return obs.values.astype(np.float32)

    def step(self, action):
        current_price = self.df.iloc[self.current_step]["Close"]
        reward = 0

        if action == 1:  # Buy
            if self.position == 0:
                self.position = 1
                self.entry_price = current_price
            elif self.position == -1:
                reward = self.entry_price - current_price
                self.balance += reward
                self.total_profit += reward
                self.position = 0
        elif action == 2:  # Sell
            if self.position == 0:
                self.position = -1
                self.entry_price = current_price
            elif self.position == 1:
                reward = current_price - self.entry_price
                self.balance += reward
                self.total_profit += reward
                self.position = 0

        self.trades.append((self.current_step, action, reward))
        self.current_step += 1
        done = self.current_step >= len(self.df)

        return self._get_observation(), reward, done, False, {}

    def render(self):
        print(f"Step: {self.current_step}, Balance: {self.balance:.2f}, Profit: {self.total_profit:.2f}, Position: {self.position}")
