import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Dense, Concatenate, Flatten, Lambda
from tensorflow.keras.optimizers import Adam
import gym
from gym import spaces
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
import time
from collections import deque

# ========== CONFIG ==========
SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "XAUUSD"]
BASE_DIR = r"C:/FPFX"
DATA_DIR = os.path.join(BASE_DIR, "Data", "processed")
NEWS_CSV = os.path.join(BASE_DIR, "Data", "news_cache.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")

NB_EVAL_EPISODES = 5
FIXED_CONFIDENCE_THRESHOLD = 0.65
INITIAL_BALANCE = 100000.0
DISCOUNT_FACTOR = 0.99
OVERTRADE_BARS = 12
MAX_OPEN_TRADES = 3
DAILY_DD_LIMIT = 0.65
MIN_RISK = 0.005      # 0.3%
MAX_RISK = 0.01      # 0.8%
MIN_LOT_SIZE = 0.1
MAX_LOT_SIZE = 15.0
COMMISSION_PIPS = 0.02
TIME_PENALTY_FACTOR = 0.1
SHARPE_WINDOW = 1000
SHARPE_EPSILON = 1e-5

# Risk Management (matches training exactly)
MIN_SL_MULT = 1.0
MAX_SL_MULT = 10.0
MIN_TP_MULT = 1.0
MAX_TP_MULT = 10.0
SLIPPAGE_RATIO = 0.3
MIN_STOP_DISTANCE_PIPS = 10

# ========== DATA LOADER ==========
def load_all_data():
    frames = []
    for sym in SYMBOLS:
        path = os.path.join(DATA_DIR, f"{sym}_processed.csv")
        df = pd.read_csv(path, parse_dates=["time"])
        df["symbol"] = sym
        frames.append(df)
    
    all_df = pd.concat(frames, ignore_index=True)
    all_df.sort_values(["time","symbol"], inplace=True)
    
    news = pd.read_csv(NEWS_CSV, parse_dates=["date"])
    if news["date"].dt.tz is None:
        news["date"] = news["date"].dt.tz_localize("UTC")
    
    dates = pd.date_range(all_df["time"].min().floor("D"),
                         all_df["time"].max().ceil("D"), freq="D")
    idx = pd.MultiIndex.from_product([dates, SYMBOLS], names=["date","symbol"])
    
    news = (news.set_index(["date","symbol"])
              .reindex(idx)
              .groupby(level="symbol").ffill()
              .reset_index())
    
    if all_df["time"].dt.tz is None:
        all_df["date"] = all_df["time"].dt.tz_localize("UTC").dt.floor("D")
    else:
        all_df["date"] = all_df["time"].dt.tz_convert("UTC").dt.floor("D")
    
    all_df = all_df.merge(news, on=["date","symbol"], how="left")
    
    all_df = all_df.assign(
        news_count=all_df["news_count"].fillna(0),
        avg_sentiment=all_df["avg_sentiment"].fillna(0.0)
    )
    
    all_df.ffill(inplace=True)
    all_df.fillna(0, inplace=True)
    return all_df

ALL_DATA = load_all_data()

# ========== MODEL BUILDERS ==========
def build_actor(input_shape, action_space):
    s = Input(shape=input_shape)
    x = Flatten()(s)
    x = Dense(256, activation="relu")(x)
    x = Dense(256, activation="relu")(x)
    x = Dense(128, activation="relu")(x)
    raw = Dense(action_space.shape[0], activation="sigmoid")(x)
    diff = action_space.high - action_space.low
    low = action_space.low
    a = Lambda(lambda y: y * diff + low)(raw)
    return tf.keras.models.Model(s, a)

def build_critic(input_shape, action_space):
    s = Input(shape=input_shape)
    a = Input(shape=(action_space.shape[0],))
    x = Flatten()(s)
    x = Concatenate()([x, a])
    x = Dense(256, activation="relu")(x)
    x = Dense(256, activation="relu")(x)
    x = Dense(128, activation="relu")(x)
    q = Dense(1, activation="linear")(x)
    return tf.keras.models.Model([s, a], q)

# ========== ENVIRONMENT ==========
class ForexMultiEnv(gym.Env):
    def __init__(self, all_data, initial_balance=INITIAL_BALANCE):
        super().__init__()
        self.all_data = all_data.copy()
        self.symbols = SYMBOLS
        self.n_rows = len(all_data)
        self.sym2id = {s:i for i,s in enumerate(self.symbols)}
        feat_cols = [c for c in all_data.columns if c not in ("time","symbol","date")]
        self.feature_count = len(feat_cols)
        
        self.observation_space = spaces.Box(
            -np.inf, np.inf,
            shape=(self.feature_count + len(self.symbols),),
            dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=np.array([0.0, MIN_SL_MULT, MIN_TP_MULT, -0.1, 0.0]),
            high=np.array([1.0, MAX_SL_MULT, MAX_TP_MULT, 0.1, 1.0]),
            dtype=np.float32
        )
        
        self.initial_balance = initial_balance
        self.scaler = StandardScaler().fit(all_data[feat_cols].values)
        self.reset()
    
    def _get_pip_size(self, symbol):
        return 0.0001 if "JPY" not in symbol else 0.01
    
    def _get_pip_value(self, symbol, price):
        if "JPY" in symbol:
            return 1000 / price
        elif symbol == "XAUUSD":
            return 10.0
        else:
            return 10.0
    
    def reset(self):
        self.balance = self.initial_balance
        self.max_balance = self.balance
        self.max_drawdown = 0.0
        self.open_positions = {}
        self.current_step = 0
        self.recent_rewards = deque(maxlen=SHARPE_WINDOW)
        return self._obs()
    
    def _obs(self):
        row = self.all_data.iloc[self.current_step]
        feats = self.scaler.transform([row.drop(["time","symbol","date"]).values])[0]
        oh = np.zeros(len(self.symbols))
        oh[self.sym2id[row["symbol"]]] = 1
        return np.concatenate([feats, oh])
    
    def step(self, action):
        row = self.all_data.iloc[self.current_step]
        symbol = row["symbol"]
        price = row["open"]
        
        signal, sl_mult, tp_mult, sent_exit, conf = np.clip(
            action,
            [0.0, MIN_SL_MULT, MIN_TP_MULT, -0.1, 0.0],
            [1.0, MAX_SL_MULT, MAX_TP_MULT, 0.1, 1.0]
        )
        conf = conf if conf >= FIXED_CONFIDENCE_THRESHOLD else 0.0
        
        self.balance = max(0, self.balance)
        self.max_balance = max(self.max_balance, self.balance)
        self.max_drawdown = max(self.max_drawdown,
                              (self.max_balance - self.balance) / self.max_balance)
        
        if self.balance < 1000 or self.max_drawdown >= DAILY_DD_LIMIT:
            print(f"\n[EARLY RESET] Bal ${self.balance:.2f} DD {self.max_drawdown:.2%}")
            return self.reset(), 0.0, True, {"balance": self.balance}
        
        reward = 0.0
        
        if conf > 0 and len(self.open_positions) < MAX_OPEN_TRADES and symbol not in self.open_positions:
            risk_amt = max(MIN_RISK * self.balance,
                          (MIN_RISK + conf * (MAX_RISK - MIN_RISK)) * self.balance)
            pip_size = self._get_pip_size(symbol)
            pip_value = self._get_pip_value(symbol, price)
            
            # Critical ATR scaling (matches training exactly)
            atr_pips = (row["ATR_14"] * 10) / pip_size  # ×10 multiplier
            
            stop_distance_pips = max(atr_pips * sl_mult, MIN_STOP_DISTANCE_PIPS)
            tp_distance_pips = atr_pips * tp_mult
            
            lot = risk_amt / (stop_distance_pips * pip_value)
            lot = max(min(lot, MAX_LOT_SIZE), MIN_LOT_SIZE)
            
            direction = 1 if signal >= 0.5 else -1
            entry_price = price
            
            if direction == 1:
                sl_price = entry_price - stop_distance_pips * pip_size
                tp_price = entry_price + tp_distance_pips * pip_size
            else:
                sl_price = entry_price + stop_distance_pips * pip_size
                tp_price = entry_price - tp_distance_pips * pip_size
            
            risk_dist = abs(entry_price - sl_price)
            reward_dist = abs(tp_price - entry_price)
            rr_ratio = reward_dist / risk_dist if risk_dist > 0 else 1.0
            
            if rr_ratio < 1.0:
                reward -= 10.0
            elif rr_ratio > 5.0:
                reward -= 5.0
            
            if lot >= MIN_LOT_SIZE:
                self.open_positions[symbol] = {
                    "direction": direction,
                    "entry_price": entry_price,
                    "sl_price": sl_price,
                    "tp_price": tp_price,
                    "lot_size": lot,
                    "sent_exit": sent_exit,
                    "confidence": conf,
                    "entry_step": self.current_step,
                    "rr_ratio": rr_ratio
                }
        
        if symbol in self.open_positions:
            pos = self.open_positions[symbol]
            exit_price = None
            reason = None
            pip_size = self._get_pip_size(symbol)
            pip_value = self._get_pip_value(symbol, pos["entry_price"])
            
            # Consistent ATR scaling for slippage
            atr_pips = (row["ATR_14"] * 10) / pip_size
            
            if ((pos["direction"] == 1 and row["low"] <= pos["sl_price"]) or
                (pos["direction"] == -1 and row["high"] >= pos["sl_price"])):
                exit_price = pos["sl_price"]
                reason = "SL"
                exit_price += -pos["direction"] * (SLIPPAGE_RATIO * atr_pips * pip_size)
            elif ((pos["direction"] == 1 and row["high"] >= pos["tp_price"]) or
                  (pos["direction"] == -1 and row["low"] <= pos["tp_price"])):
                exit_price = pos["tp_price"]
                reason = "TP"
            elif ((pos["direction"] == 1 and row["avg_sentiment"] <= pos["sent_exit"]) or
                  (pos["direction"] == -1 and row["avg_sentiment"] >= pos["sent_exit"])):
                exit_price = row["close"]
                reason = "SENT"
            
            if exit_price is not None:
                pnl = (exit_price - pos["entry_price"]) * pos["direction"] * pos["lot_size"] * (100000 if "XAU" not in symbol else 100)
                if "JPY" in symbol:
                    pnl = (exit_price - pos["entry_price"]) * pos["direction"] * pos["lot_size"] * 1000 / exit_price
                pnl -= COMMISSION_PIPS * pos["lot_size"] * pip_value
                self.balance += pnl
                reward = pnl * pos["rr_ratio"] - (self.current_step - pos["entry_step"]) * TIME_PENALTY_FACTOR
                if pnl < 0:
                    reward *= (1 + pos["confidence"])
                
                del self.open_positions[symbol]
        
        self.current_step += 1
        done = self.current_step >= self.n_rows
        
        self.recent_rewards.append(reward)
        if len(self.recent_rewards) >= 2:
            mean_r = np.mean(self.recent_rewards)
            std_r = np.std(self.recent_rewards)
            if std_r < SHARPE_EPSILON:
                std_r = SHARPE_EPSILON
            reward = (reward - mean_r) / std_r
        
        return self._obs(), reward, done, {
            "balance": self.balance,
            "max_drawdown": self.max_drawdown
        }

# ========== MAIN ==========
if __name__ == "__main__":
    from rl.callbacks import Callback
    
    class EvalLogger(Callback):
        def __init__(self, initial_balance=INITIAL_BALANCE):
            self.initial_balance = initial_balance
            self.episode_rewards = []
            self.final_balances = []
            self.episode_drawdowns = []
        
        def on_episode_end(self, episode, logs={}):
            rew = logs.get("episode_reward", 0)
            info = logs.get("info", {})
            bal = info.get("balance", self.initial_balance)
            dd = info.get("max_drawdown", 0)
            
            self.episode_rewards.append(rew)
            self.final_balances.append(bal)
            self.episode_drawdowns.append(dd)
            
            print(f"[Episode {episode+1}] Reward: {rew:.2f}, Final Balance: {bal:.2f}, Max DD: {dd:.2%}")
        
        def on_train_end(self, logs=None):
            avg_reward = np.mean(self.episode_rewards)
            avg_balance = np.mean(self.final_balances)
            avg_dd = np.mean(self.episode_drawdowns)
            
            print(f"\n=== Evaluation Summary ===")
            print(f"Initial Balance: ${self.initial_balance:,.2f}")
            print(f"Avg Reward: {avg_reward:.2f}")
            print(f"Avg Final Balance: {avg_balance:,.2f}")
            print(f"Avg Max DD: {avg_dd:.2%}")
            print(f"Min Balance: ${min(self.final_balances):,.2f}")
            print(f"Max Balance: ${max(self.final_balances):,.2f}")
            print(f"ROI: {(avg_balance - self.initial_balance)/self.initial_balance:.2%}")
    
    env = ForexMultiEnv(ALL_DATA)
    n_actions = env.action_space.shape[0]
    
    # Build models
    actor = build_actor((1,) + env.observation_space.shape, env.action_space)
    critic = build_critic((1,) + env.observation_space.shape, env.action_space)
    
    # Load trained models with proper compilation
    try:
        print("Loading actor.h5 ...")
        actor_path = os.path.join(MODEL_DIR, "actor.h5")
        actor = load_model(actor_path)
        actor.compile(optimizer=Adam(1e-4), loss='mse')  # Critical fix
        print(f"Successfully loaded and compiled actor from {actor_path}")
        
        print("Restoring critic checkpoint ...")
        checkpoint = tf.train.Checkpoint(critic=critic)
        ckpt = tf.train.latest_checkpoint(MODEL_DIR)
        if ckpt is None: 
            raise FileNotFoundError(f"No checkpoint found in {MODEL_DIR}")
        print(f"→ Found checkpoint: {ckpt}")
        status = checkpoint.restore(ckpt)
        status.expect_partial()
        print("Critic weights restored successfully")
    except Exception as e:
        print(f"Error loading models: {e}")
        exit(1)
    
    # Configure agent
    memory = SequentialMemory(limit=1, window_length=1)
    rnd = OrnsteinUhlenbeckProcess(size=n_actions, theta=0.15, mu=0.0, sigma=0.2, sigma_min=0.05)
    
    agent = DDPGAgent(
        nb_actions=n_actions, actor=actor, critic=critic,
        critic_action_input=critic.input[1],
        memory=memory, nb_steps_warmup_actor=0,
        nb_steps_warmup_critic=0, random_process=rnd,
        gamma=DISCOUNT_FACTOR, target_model_update=1e-3, batch_size=64
    )
    
    agent.compile(Adam(1e-4), metrics=["mae"])
    
    print(f"Testing {NB_EVAL_EPISODES} episodes (conf ≥ {FIXED_CONFIDENCE_THRESHOLD})")
    agent.test(env, nb_episodes=NB_EVAL_EPISODES, visualize=False, callbacks=[EvalLogger()])