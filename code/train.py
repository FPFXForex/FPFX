import os
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, Flatten
import tensorflow as tf
import gym
from gym import spaces
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from tensorflow.keras.optimizers.legacy import Adam
import time

# ========== CONFIGURATION ==========
SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "XAUUSD"]
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "Data", "Processed")
NEWS_CSV = os.path.join(BASE_DIR, "Data", "news_cache.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

# Hyperparameters
INITIAL_BALANCE = 100000.0
DISCOUNT_FACTOR = 0.99
TRAIN_STEPS = 200000
MEMORY_LIMIT = 1000000
BATCH_SIZE = 64
OVERTRADE_BARS = 12
MAX_OPEN_TRADES = 3
DAILY_DD_LIMIT = 1.0
MIN_RISK = 0.01
MAX_RISK = 0.02
FIXED_CONFIDENCE_THRESHOLD = 0.001

# ========== DATA LOADING ==========
def load_all_data():
    print("\n[1/4] Loading market data...")
    frames = []
    for sym in SYMBOLS:
        path = os.path.join(DATA_DIR, f"{sym}_processed.csv")
        df = pd.read_csv(path, parse_dates=["time"])
        df["ATR_14"] = df["ATR_14"] * 10
        if "KC_upper" in df.columns:
            df["KC_upper"] = df["KC_upper"] * 10
            df["KC_lower"] = df["KC_lower"] * 10
        df["symbol"] = sym
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    all_df.sort_values(by=["time", "symbol"], inplace=True)

    print("[2/4] Merging news sentiment...")
    try:
        news = pd.read_csv(NEWS_CSV, parse_dates=["date"])
        if news["date"].dt.tz is None:
            news["date"] = news["date"].dt.tz_localize('UTC')

        all_dates = pd.date_range(
            start=all_df["time"].min().floor("D"),
            end=all_df["time"].max().ceil("D"),
            freq="D"
        )
        full_index = pd.MultiIndex.from_product(
            [all_dates, SYMBOLS],
            names=["date", "symbol"]
        )
        news = (news.set_index(["date", "symbol"])
                .reindex(full_index)
                .groupby(level="symbol").ffill()
                .reset_index())
    except Exception as e:
        print(f"[WARNING] News loading failed: {str(e)}")
        news = pd.DataFrame(columns=["date", "symbol", "news_count", "avg_sentiment"])

    if all_df["time"].dt.tz is None:
        all_df["date"] = all_df["time"].dt.tz_localize('UTC').dt.floor("D")
    else:
        all_df["date"] = all_df["time"].dt.tz_convert('UTC').dt.floor("D")

    all_df = all_df.merge(news, how="left", on=["date", "symbol"])
    all_df["news_count"] = all_df["news_count"].fillna(0)
    all_df["avg_sentiment"] = all_df["avg_sentiment"].fillna(0.0)
    return all_df

ALL_DATA = load_all_data()

# ========== ENVIRONMENT ==========
class ForexMultiEnv(gym.Env):
    def __init__(self, all_data, initial_balance=INITIAL_BALANCE):
        super(ForexMultiEnv, self).__init__()
        self.all_data = all_data.copy()
        self.symbols = SYMBOLS
        self.n_rows = len(self.all_data)
        self.symbol_to_id = {sym: i for i, sym in enumerate(SYMBOLS)}
        feature_cols = [c for c in self.all_data.columns if c not in ["time", "symbol", "date"]]
        self.feature_count = len(feature_cols)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.feature_count + len(SYMBOLS),),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=np.array([0.0, 1.0, 1.0, -0.1, 0.0]),
            high=np.array([1.0, 10.0, 10.0, 0.1, 1.0]),
            dtype=np.float32
        )

        self.initial_balance = initial_balance

        # scaler initialized before reset
        self.features = self.all_data[feature_cols].values
        self.scaler = StandardScaler()
        self.scaler.fit(self.features)

        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.max_drawdown = 0.0
        self.open_positions = {}
        self.last_entry_step = {sym: -OVERTRADE_BARS - 1 for sym in self.symbols}
        self.current_step = 0
        self.episode_trades = 0
        self.total_trades = 0
        self.force_trade_counter = 0
        return self._get_observation(self.current_step)

    def step(self, action):
        row = self.all_data.iloc[self.current_step]
        symbol = row["symbol"]

        if row["high"] <= row["low"]:
            self.current_step += 1
            return self._get_observation(self.current_step), 0, False, {}

        signal, sl_mult, tp_mult, sent_exit_thresh, confidence = np.clip(
            action, self.action_space.low, self.action_space.high
        )

        self.force_trade_counter += 1
        if self.total_trades == 0 and self.force_trade_counter >= 3000:
            confidence = max(confidence, 0.5)
            print(f"\n!!! FORCING FIRST TRADE AT STEP {self.current_step} !!!")

        if (confidence >= FIXED_CONFIDENCE_THRESHOLD and
            len(self.open_positions) < MAX_OPEN_TRADES and
            symbol not in self.open_positions):

            atr = row["ATR_14"]
            pip_value = 0.01 if "JPY" in symbol or symbol == "XAUUSD" else 0.0001
            risk_amount = (MIN_RISK + confidence * (MAX_RISK - MIN_RISK)) * self.balance
            lot_size = max(risk_amount / (atr * sl_mult / pip_value), 0.01)
            direction = 1 if signal >= 0.5 else -1

            self.open_positions[symbol] = {
                "direction": direction,
                "entry_price": row["open"],
                "sl_price": row["open"] - direction * atr * sl_mult,
                "tp_price": row["open"] + direction * atr * tp_mult,
                "lot_size": lot_size,
                "entry_step": self.current_step,
                "sent_exit_thresh": sent_exit_thresh
            }
            self.episode_trades += 1
            self.total_trades += 1
            print(f"\n[TRADE] {'LONG' if direction==1 else 'SHORT'} {symbol} "
                  f"@{row['open']:.5f} (Size: {lot_size:.2f} lots, "
                  f"ATR: {atr:.5f}, SL: {atr*sl_mult:.5f})")

        reward = 0
        if symbol in self.open_positions:
            pos = self.open_positions[symbol]
            exit_price, exit_reason = None, None

            if ((pos["direction"] == 1 and row["low"] <= pos["sl_price"]) or
                (pos["direction"] == -1 and row["high"] >= pos["sl_price"])):
                exit_price = pos["sl_price"]
                exit_reason = "SL"
            elif ((pos["direction"] == 1 and row["high"] >= pos["tp_price"]) or
                  (pos["direction"] == -1 and row["low"] <= pos["tp_price"])):
                exit_price = pos["tp_price"]
                exit_reason = "TP"
            elif ((pos["direction"] == 1 and row["avg_sentiment"] <= pos["sent_exit_thresh"]) or
                  (pos["direction"] == -1 and row["avg_sentiment"] >= pos["sent_exit_thresh"])):
                exit_price = row["close"]
                exit_reason = "SENT"

            if exit_price:
                pip_value = 0.01 if "JPY" in symbol or symbol == "XAUUSD" else 0.0001
                pnl = (exit_price - pos["entry_price"]) * pos["direction"] * (pos["lot_size"] / pip_value)
                self.balance += pnl
                reward = pnl
                del self.open_positions[symbol]
                print(f"[EXIT] {symbol} @ {exit_price:.5f} ({exit_reason}) "
                      f"| PnL: ${pnl:+.2f}")

        self.current_step += 1
        done = self.current_step >= self.n_rows
        next_obs = np.zeros_like(self.observation_space.low) if done else self._get_observation(self.current_step)

        if self.current_step % 5000 == 0:
            print(f"\n[Progress] Step: {self.current_step}/{self.n_rows} | "
                  f"Balance: ${self.balance:,.2f} | "
                  f"Trades: {self.total_trades}")

        return next_obs, reward, done, {"balance": self.balance}

    def _get_observation(self, step):
        row = self.all_data.iloc[step]
        features = self.scaler.transform([row.drop(["time", "symbol", "date"]).values])[0]
        symbol_onehot = np.zeros(len(SYMBOLS))
        symbol_onehot[self.symbol_to_id[row["symbol"]]] = 1
        return np.concatenate([features, symbol_onehot])

# ========== MODEL ARCHITECTURE ==========
def build_actor(input_shape, action_space):
    state_input = Input(shape=input_shape, name="state_input")
    x = Flatten()(state_input)
    x = Dense(256, activation="relu")(x)
    x = Dense(256, activation="relu")(x)
    x = Dense(128, activation="relu")(x)
    raw_actions = Dense(action_space.shape[0], activation="sigmoid", name="raw_actions")(x)
    actions = tf.keras.layers.Lambda(
        lambda x: x * (action_space.high - action_space.low) + action_space.low,
        name="actions"
    )(raw_actions)
    return Model(inputs=state_input, outputs=actions)

def build_critic(input_shape, action_space):
    state_input = Input(shape=input_shape, name="state_input")
    x = Flatten()(state_input)
    action_input = Input(shape=(action_space.shape[0],), name="action_input")
    x = Concatenate()([x, action_input])
    x = Dense(256, activation="relu")(x)
    x = Dense(256, activation="relu")(x)
    x = Dense(128, activation="relu")(x)
    q_value = Dense(1, activation="linear", name="q_value")(x)
    return Model(inputs=[state_input, action_input], outputs=q_value)

# ========== TRAINING ==========
def train_agent():
    env = ForexMultiEnv(ALL_DATA)
    nb_actions = env.action_space.shape[0]

    actor = build_actor((1,) + env.observation_space.shape, env.action_space)
    critic = build_critic((1,) + env.observation_space.shape, env.action_space)

    memory = SequentialMemory(limit=MEMORY_LIMIT, window_length=1)
    random_process = OrnsteinUhlenbeckProcess(
        size=nb_actions, theta=0.3, mu=0.0, sigma=0.4, sigma_min=0.1
    )

    agent = DDPGAgent(
        nb_actions=nb_actions,
        actor=actor,
        critic=critic,
        critic_action_input=critic.input[1],
        memory=memory,
        nb_steps_warmup_actor=10000,
        nb_steps_warmup_critic=10000,
        random_process=random_process,
        gamma=DISCOUNT_FACTOR,
        target_model_update=1e-3,
        batch_size=BATCH_SIZE
    )
    agent.compile(Adam(learning_rate=1e-4), metrics=["mae"])

    print("\nStarting training...")
    agent.fit(env, nb_steps=TRAIN_STEPS, visualize=False, verbose=1)

    actor.save(os.path.join(MODEL_DIR, "actor.h5"))
    agent.save_weights(os.path.join(MODEL_DIR, "ddpg_weights.h5f"), overwrite=True)

if __name__ == "__main__":
    train_agent()
