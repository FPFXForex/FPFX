import os
import random
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, Flatten, Lambda
import tensorflow as tf
import gym
from gym import spaces
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from tensorflow.keras.optimizers import Adam
import time
from rl.callbacks import Callback

# ========== CONFIGURATION ==========
SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "XAUUSD"]
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "Data", "Processed")
NEWS_CSV = os.path.join(BASE_DIR, "Data", "news_cache.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Hyperparameters
INITIAL_BALANCE = 100000.0
DISCOUNT_FACTOR = 0.99
TRAIN_STEPS = 700000
MEMORY_LIMIT = 1000000
BATCH_SIZE = 64
OVERTRADE_BARS = 12
MAX_OPEN_TRADES = 3
DAILY_DD_LIMIT = 0.2  # 20% max daily drawdown
MIN_RISK = 0.005  # 0.5% per trade
MAX_RISK = 0.02   # 2% per trade
FIXED_CONFIDENCE_THRESHOLD = 0.0  # exploration during training
MIN_LOT_SIZE = 0.1
MAX_LOT_SIZE = 20.0  # Increased maximum lot size to 20

# ========== DATA LOADING ==========
def load_all_data():
    print("\n[1/4] Loading market data...")
    frames = []
    
    for sym in SYMBOLS:
        path = os.path.join(DATA_DIR, f"{sym}_processed.csv")
        try:
            df = pd.read_csv(path, parse_dates=["time"])
            df["ATR_14"] = df["ATR_14"] * 10
            if "KC_upper" in df.columns:
                df["KC_upper"] = df["KC_upper"] * 10
                df["KC_lower"] = df["KC_lower"] * 10
            df["symbol"] = sym
            frames.append(df)
        except Exception as e:
            print(f"Error loading {sym}: {str(e)}")
            continue

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

    all_df.fillna(method="ffill", inplace=True)
    all_df.fillna(0.0, inplace=True)
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
            low=np.array([0.0, 1.0, 1.0, -0.1, 0.0], dtype=np.float32),
            high=np.array([1.0, 10.0, 10.0, 0.1, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        self.initial_balance = initial_balance
        self.current_episode = 0
        
        self.features = self.all_data[feature_cols].values
        self.scaler = StandardScaler()
        self.scaler.fit(self.features)
        
        self.forced_first_trade = False
        self.first_real_trade_logged = False
        self.recent_trades = []
        self.recent_exits = []
        self.last_summary_time = time.time()
        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.max_balance = self.initial_balance
        self.max_drawdown = 0.0
        self.open_positions = {}
        self.last_entry_step = {sym: -OVERTRADE_BARS - 1 for sym in self.symbols}
        self.current_step = 0
        self.episode_trades = 0
        self.total_trades = 0
        self.force_trade_counter = 0
        self.forced_first_trade = False
        self.recent_trades.clear()
        self.recent_exits.clear()
        self.current_episode += 1
        return self._get_observation(self.current_step)

    def step(self, action):
        row = self.all_data.iloc[self.current_step]
        symbol = row["symbol"]
        
        if row["high"] <= row["low"]:
            self.current_step += 1
            return self._get_observation(self.current_step), 0, False, {"balance": self.balance}
        
        signal, sl_mult, tp_mult, sent_exit_thresh, confidence = np.clip(
            action, self.action_space.low, self.action_space.high
        )
        
        self.balance = max(0, self.balance)
        self.max_balance = max(self.max_balance, self.balance)
        self.max_drawdown = max(self.max_drawdown, (self.max_balance - self.balance) / self.max_balance)
        
        if self.balance < 1000 or self.max_drawdown >= DAILY_DD_LIMIT:
            print(f"\n[EARLY RESET] Episode {self.current_episode} | Balance: ${self.balance:.2f} | Drawdown: {self.max_drawdown:.2%}")
            return self.reset(), 0, True, {"balance": self.balance, "early_reset": True, "max_drawdown": self.max_drawdown}
        
        self.force_trade_counter += 1
        if (not self.forced_first_trade and self.total_trades == 0 
            and self.force_trade_counter >= 3000):
            confidence = max(confidence, 0.5)
            self.forced_first_trade = True
        
        reward = 0
        
        # TRADE ENTRY
        if (confidence >= FIXED_CONFIDENCE_THRESHOLD and
            len(self.open_positions) < MAX_OPEN_TRADES and
            symbol not in self.open_positions):
            
            atr = row["ATR_14"]
            pip_value = 0.01 if "JPY" in symbol or symbol == "XAUUSD" else 0.0001
            
            risk_amount = max(
                MIN_RISK * self.balance,
                (MIN_RISK + confidence * (MAX_RISK - MIN_RISK)) * self.balance
            )
            
            lot_size = max(min(
                risk_amount / (atr * sl_mult / pip_value),
                MAX_LOT_SIZE),
                MIN_LOT_SIZE
            )
            
            if lot_size < MIN_LOT_SIZE or lot_size > MAX_LOT_SIZE:
                self.current_step += 1
                return self._get_observation(self.current_step), 0, False, {"balance": self.balance}
            
            direction = 1 if signal >= 0.5 else -1
            
            self.open_positions[symbol] = {
                "direction": direction,
                "entry_price": row["open"],
                "sl_price": row["open"] - direction * atr * sl_mult,
                "tp_price": row["open"] + direction * atr * tp_mult,
                "lot_size": lot_size,
                "entry_step": self.current_step,
                "sent_exit_thresh": sent_exit_thresh,
                "confidence": confidence
            }
            
            self.episode_trades += 1
            self.total_trades += 1
            
            self.recent_trades.append({
                "symbol": symbol,
                "direction": "LONG" if direction == 1 else "SHORT",
                "entry_price": row["open"],
                "sl_price": row["open"] - direction * atr * sl_mult,
                "tp_price": row["open"] + direction * atr * tp_mult,
                "lot_size": lot_size,
                "ATR": atr,
                "confidence": confidence
            })
        
        # TRADE EXIT
        if symbol in self.open_positions:
            pos = self.open_positions[symbol]
            exit_price, exit_reason = None, None
            
            if ((pos["direction"] == 1 and row["low"] <= pos["sl_price"]) or
                (pos["direction"] == -1 and row["high"] >= pos["sl_price"])):
                exit_price = pos["sl_price"]
                exit_reason = "SL"
                reward = -abs(pos["lot_size"] * 2)
            
            elif ((pos["direction"] == 1 and row["high"] >= pos["tp_price"]) or
                  (pos["direction"] == -1 and row["low"] <= pos["tp_price"])):
                exit_price = pos["tp_price"]
                exit_reason = "TP"
                reward = pos["lot_size"]
            
            elif ((pos["direction"] == 1 and row["avg_sentiment"] <= pos["sent_exit_thresh"]) or
                  (pos["direction"] == -1 and row["avg_sentiment"] >= pos["sent_exit_thresh"])):
                exit_price = row["close"]
                exit_reason = "SENT"
                reward = 0
            
            if exit_price is not None:
                pip_value = 0.01 if "JPY" in symbol or symbol == "XAUUSD" else 0.0001
                pnl = (exit_price - pos["entry_price"]) * pos["direction"] * (pos["lot_size"] / pip_value)
                self.balance += pnl
                
                self.recent_exits.append({
                    "symbol": symbol,
                    "exit_price": exit_price,
                    "reason": exit_reason,
                    "pnl": pnl,
                    "confidence": pos["confidence"]
                })
                
                del self.open_positions[symbol]
        
        self.current_step += 1
        done = self.current_step >= self.n_rows
        
        next_obs = np.zeros_like(self.observation_space.low) if done else self._get_observation(self.current_step)
        
        # aggregated summary every 30 seconds
        if time.time() - self.last_summary_time >= 30:
            entry_count = len(self.recent_trades)
            exit_count = len(self.recent_exits)
            avg_entry_conf = np.mean([t["confidence"] for t in self.recent_trades]) if entry_count > 0 else 0.0
            avg_exit_conf = np.mean([e["confidence"] for e in self.recent_exits]) if exit_count > 0 else 0.0
            total_exit_pnl = sum(e["pnl"] for e in self.recent_exits)
            print(f"\n[SUMMARY] Episode {self.current_episode} | Step {self.current_step}/{self.n_rows} "
                  f"| Balance: ${self.balance:,.2f} | Drawdown: {self.max_drawdown:.2%} | Total Trades: {self.total_trades}")
            print(f" Entries placed: {entry_count} | Avg Entry Conf: {avg_entry_conf:.4f}")
            print(f" Exits executed: {exit_count} | Exit PnL: ${total_exit_pnl:+.2f} | Avg Exit Conf: {avg_exit_conf:.4f}")
            self.recent_trades.clear()
            self.recent_exits.clear()
            self.last_summary_time = time.time()
        
        return next_obs, reward, done, {"balance": self.balance, "max_drawdown": self.max_drawdown}

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
    
    diff = action_space.high - action_space.low
    low = action_space.low
    actions = Lambda(lambda x, d=diff, l=low: x * d + l, name="actions")(raw_actions)
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

# ========== CHECKPOINT CALLBACK ==========
class CheckpointSaver(Callback):
    def __init__(self, save_path, interval=50000):
        super().__init__()
        self.save_path = save_path
        self.interval = interval
        os.makedirs(self.save_path, exist_ok=True)

    def on_step_end(self, step, logs={}):
        if step % self.interval == 0 and step > 0:
            filename = os.path.join(self.save_path, f"ddpg_weights_step_{step}.h5f")
            self.model.save_weights(filename, overwrite=True)
            print(f"\n[SAVED] Weights checkpoint at step {step} ➝ {filename}")

def get_latest_checkpoint(path):
    checkpoints = glob.glob(os.path.join(path, "ddpg_weights_step_*.h5f"))
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getmtime)

# ========== TRAINING ==========
def train_agent():
    env = ForexMultiEnv(ALL_DATA)
    nb_actions = env.action_space.shape[0]
    
    actor = build_actor((1,) + env.observation_space.shape, env.action_space)
    critic = build_critic((1,) + env.observation_space.shape, env.action_space)
    
    memory = SequentialMemory(limit=MEMORY_LIMIT, window_length=1)
    random_process = OrnsteinUhlenbeckProcess(
        size=nb_actions, theta=0.15, mu=0.0, sigma=0.2, sigma_min=0.05
    )
    
    agent = DDPGAgent(
        nb_actions=nb_actions,
        actor=actor,
        critic=critic,
        critic_action_input=critic.input[1],
        memory=memory,
        nb_steps_warmup_actor=1000,
        nb_steps_warmup_critic=1000,
        random_process=random_process,
        gamma=DISCOUNT_FACTOR,
        target_model_update=1e-3,
        batch_size=BATCH_SIZE
    )
    
    agent.compile(Adam(learning_rate=1e-4), metrics=["mae"])
    
    # Attempt to resume from latest checkpoint
    latest = get_latest_checkpoint(CHECKPOINT_DIR)
    if latest:
        print(f"\n[RESUMING] Loading weights from: {latest}")
        agent.load_weights(latest)
    
    print("\nStarting training...")
    callbacks = [CheckpointSaver(save_path=CHECKPOINT_DIR, interval=50000)]
    agent.fit(env, nb_steps=TRAIN_STEPS, visualize=False, verbose=0, callbacks=callbacks)
    
    # ========== EVALUATION ==========
    print("\n=== EVALUATING LEARNED POLICY ===")
    global FIXED_CONFIDENCE_THRESHOLD
    FIXED_CONFIDENCE_THRESHOLD = 0.5  # only trades with confidence ≥ 0.5 now
    
    eval_env = ForexMultiEnv(ALL_DATA)
    eval_env.reset()
    agent.test(eval_env, nb_episodes=5, visualize=False)
    
    actor.save(os.path.join(MODEL_DIR, "actor.h5"))
    agent.save_weights(os.path.join(MODEL_DIR, "ddpg_weights.h5f"), overwrite=True)
    
    stats = {
        "total_episodes": eval_env.current_episode,
        "final_balance": eval_env.balance,
        "total_trades": eval_env.total_trades,
        "max_drawdown": eval_env.max_drawdown
    }
    joblib.dump(stats, os.path.join(MODEL_DIR, "training_stats.pkl"))

if __name__ == "__main__":
    train_agent()
