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
from collections import deque

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
DAILY_DD_LIMIT = 0.2
MIN_RISK = 0.005
MAX_RISK = 0.02
FIXED_CONFIDENCE_THRESHOLD = 0.0
MIN_LOT_SIZE = 0.1
MAX_LOT_SIZE = 20.0
COMMISSION_PIPS = 0.02
TIME_PENALTY_FACTOR = 0.1
OPPORTUNITY_PENALTY = 10.0
SHARPE_WINDOW = 1000
SHARPE_EPSILON = 1e-5
MIN_SL_MULT = 1.0
MAX_SL_MULT = 5.0
MIN_TP_MULT = 1.0
MAX_TP_MULT = 5.0
SLIPPAGE_RATIO = 0.3

# ========== UTILITY FUNCTIONS ==========
def get_latest_checkpoint(checkpoint_dir):
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.h5f"))
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getctime)

class CheckpointSaver(Callback):
    def __init__(self, checkpoint_dir, interval=10000):
        self.checkpoint_dir = checkpoint_dir
        self.interval = interval
        os.makedirs(checkpoint_dir, exist_ok=True)
        
    def on_step_end(self, step, logs):
        if step % self.interval == 0:
            filepath = os.path.join(self.checkpoint_dir, f'checkpoint_{step}.h5f')
            self.model.save_weights(filepath, overwrite=True)

# ========== DATA LOADING ==========
def load_all_data():
    print("\n[1/4] Loading market data...")
    frames = []
    for sym in SYMBOLS:
        path = os.path.join(DATA_DIR, f"{sym}_processed.csv")
        try:
            df = pd.read_csv(path, parse_dates=["time"])
            if sym != "XAUUSD":
                df["ATR_14"] = df["ATR_14"] * 10000  # Convert to pips (0.0001 -> 1 pip)
            if "KC_upper" in df.columns:
                df["KC_upper"] = df["KC_upper"] * 10000
                df["KC_lower"] = df["KC_lower"] * 10000
            df["symbol"] = sym
            frames.append(df)
        except Exception as e:
            print(f"Error loading {sym}: {e}")
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
        full_index = pd.MultiIndex.from_product([all_dates, SYMBOLS], names=["date", "symbol"])
        news = (news.set_index(["date", "symbol"])
                .reindex(full_index)
                .groupby(level="symbol").ffill()
                .reset_index())
    except Exception as e:
        print(f"[WARNING] News loading failed: {e}")
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
            low=np.array([0.0, MIN_SL_MULT, MIN_TP_MULT, -0.1, 0.0], dtype=np.float32),
            high=np.array([1.0, MAX_SL_MULT, MAX_TP_MULT, 0.1, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        self.initial_balance = initial_balance
        self.current_episode = 0
        self.features = self.all_data[feature_cols].values
        self.scaler = StandardScaler().fit(self.features)
        self.forced_first_trade = False
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
        self.missed_opportunities = []
        self.recent_rewards = deque(maxlen=SHARPE_WINDOW)
        return self._get_observation(self.current_step)

    def step(self, action):
        row = self.all_data.iloc[self.current_step]
        symbol = row["symbol"]
        atr_pips = row["ATR_14"]  # Now in pips (already scaled in load_all_data)
        
        if row["high"] <= row["low"]:
            self.current_step += 1
            return self._get_observation(self.current_step), 0, False, {"balance": self.balance}
        
        # Clip SL/TP multipliers
        signal, sl_mult, tp_mult, sent_exit_thresh, confidence = np.clip(
            action,
            [0.0, MIN_SL_MULT, MIN_TP_MULT, -0.1, 0.0],
            [1.0, MAX_SL_MULT, MAX_TP_MULT, 0.1, 1.0]
        )
        
        # Update drawdown
        self.balance = max(0, self.balance)
        self.max_balance = max(self.max_balance, self.balance)
        self.max_drawdown = max(self.max_drawdown,
                              (self.max_balance - self.balance) / self.max_balance)
        
        # Early reset conditions
        if self.balance < 1000 or self.max_drawdown >= DAILY_DD_LIMIT:
            print(f"\n[EARLY RESET] Ep {self.current_episode} | Bal ${self.balance:.2f} | DD {self.max_drawdown:.2%}")
            return self.reset(), 0, True, {"balance": self.balance}
        
        # Forced first trade
        self.force_trade_counter += 1
        if not self.forced_first_trade and self.total_trades == 0 and self.force_trade_counter >= 3000:
            confidence = max(confidence, 0.5)
            self.forced_first_trade = True
            
        reward = 0
        
        # TRADE ENTRY
        if confidence >= FIXED_CONFIDENCE_THRESHOLD and len(self.open_positions) < MAX_OPEN_TRADES and symbol not in self.open_positions:
            # Calculate risk amount (0.5% to 2% of balance)
            risk_amt = max(MIN_RISK * self.balance,
                         (MIN_RISK + confidence * (MAX_RISK - MIN_RISK)) * self.balance)
            
            # Calculate stop distance in pips
            stop_distance_pips = atr_pips * sl_mult
            
            # Calculate proper lot size
            if "JPY" in symbol or symbol == "XAUUSD":
                lot = (risk_amt / stop_distance_pips) * 100  # For JPY and Gold (1 pip = 0.01)
            else:
                lot = risk_amt / stop_distance_pips  # For other pairs (1 pip = 0.0001)
            
            lot = max(min(lot, MAX_LOT_SIZE), MIN_LOT_SIZE)
            
            # Calculate risk-reward ratio
            direction = 1 if signal >= 0.5 else -1
            entry_price = row["open"]
            sl_price = entry_price - direction * (stop_distance_pips * (0.0001 if "JPY" not in symbol and symbol != "XAUUSD" else 0.01))
            tp_price = entry_price + direction * (atr_pips * tp_mult * (0.0001 if "JPY" not in symbol and symbol != "XAUUSD" else 0.01))
            
            risk_dist = abs(entry_price - sl_price)
            reward_dist = abs(tp_price - entry_price)
            rr_ratio = reward_dist / risk_dist if risk_dist > 0 else 1.0
            
            if rr_ratio < 1.0:
                reward -= 10.0  # Penalty for poor RR
                
            if lot >= MIN_LOT_SIZE:
                entry = {
                    "direction": direction,
                    "entry_price": entry_price,
                    "sl_price": sl_price,
                    "tp_price": tp_price,
                    "lot_size": lot,
                    "sent_exit_thresh": sent_exit_thresh,
                    "confidence": confidence,
                    "entry_step": self.current_step,
                    "rr_ratio": rr_ratio
                }
                self.open_positions[symbol] = entry
                self.episode_trades += 1
                self.total_trades += 1
                self.recent_trades.append({**entry, "symbol": symbol})
        
        # TRADE EXIT
        if symbol in self.open_positions:
            pos = self.open_positions[symbol]
            exit_price, exit_reason = None, None
            pip_value = 0.01 if "JPY" in symbol or symbol == "XAUUSD" else 0.0001
            
            # Check exit conditions
            if ((pos["direction"] == 1 and row["low"] <= pos["sl_price"]) or
                (pos["direction"] == -1 and row["high"] >= pos["sl_price"])):
                exit_price = pos["sl_price"]
                exit_reason = "SL"
                slippage = SLIPPAGE_RATIO * (atr_pips * (0.0001 if "JPY" not in symbol and symbol != "XAUUSD" else 0.01))
                exit_price += -pos["direction"] * slippage
                
            elif ((pos["direction"] == 1 and row["high"] >= pos["tp_price"]) or
                  (pos["direction"] == -1 and row["low"] <= pos["tp_price"])):
                exit_price = pos["tp_price"]
                exit_reason = "TP"
                
            elif ((pos["direction"] == 1 and row["avg_sentiment"] <= pos["sent_exit_thresh"]) or
                  (pos["direction"] == -1 and row["avg_sentiment"] >= pos["sent_exit_thresh"])):
                exit_price = row["close"]
                exit_reason = "SENT"
                
            if exit_price is not None:
                pnl = (exit_price - pos["entry_price"]) * pos["direction"] * (pos["lot_size"] / pip_value)
                commission = COMMISSION_PIPS * pos["lot_size"]
                pnl -= commission
                self.balance += pnl
                
                # Calculate reward
                reward = pnl * pos["rr_ratio"]  # Scale by risk-reward ratio
                reward -= (self.current_step - pos["entry_step"]) * TIME_PENALTY_FACTOR
                
                if pnl < 0:
                    reward *= (1 + pos["confidence"])  # Penalty for high-confidence losses
                    
                self.recent_exits.append({
                    "symbol": symbol,
                    "reason": exit_reason,
                    "entry_price": pos["entry_price"],
                    "sl_price": pos["sl_price"],
                    "tp_price": pos["tp_price"],
                    "lot_size": pos["lot_size"],
                    "confidence": pos["confidence"],
                    "exit_price": exit_price,
                    "pnl": pnl
                })
                del self.open_positions[symbol]
        
        self.current_step += 1
        done = self.current_step >= self.n_rows
        
        # Sharpe-like reward normalization
        self.recent_rewards.append(reward)
        if len(self.recent_rewards) >= 2:
            mean_r = np.mean(self.recent_rewards)
            std_r = np.std(self.recent_rewards)
            if std_r < SHARPE_EPSILON:
                std_r = SHARPE_EPSILON
            reward = (reward - mean_r) / std_r
        
        # Print summary every 2 minutes
        if time.time() - self.last_summary_time >= 120:
            self.last_summary_time = time.time()
            print(f"\n[SUMMARY] Ep {self.current_episode} | Step {self.current_step}/{self.n_rows} | Bal ${self.balance:,.2f} | DD {self.max_drawdown:.2%} | Trades {self.total_trades}")
            
            if self.open_positions:
                print("\n=== OPEN TRADES ===")
                for sym, trade in self.open_positions.items():
                    print(f"{sym}: Direction {'LONG' if trade['direction'] == 1 else 'SHORT'} | "
                          f"Entry: {trade['entry_price']:.5f} | "
                          f"SL: {trade['sl_price']:.5f} | "
                          f"TP: {trade['tp_price']:.5f} | "
                          f"Lot: {trade['lot_size']:.2f} | "
                          f"Confidence: {trade['confidence']:.2f}")
            
            if self.recent_trades:
                print("\n=== RECENT ENTRIES ===")
                for trade in self.recent_trades[-5:]:
                    print(f"{trade['symbol']}: {'LONG' if trade['direction'] == 1 else 'SHORT'} | "
                          f"Entry: {trade['entry_price']:.5f} | "
                          f"SL: {trade['sl_price']:.5f} | "
                          f"TP: {trade['tp_price']:.5f} | "
                          f"Lot: {trade['lot_size']:.2f} | "
                          f"Confidence: {trade['confidence']:.2f}")
            
            if self.recent_exits:
                print("\n=== RECENT EXITS ===")
                for trade in self.recent_exits[-5:]:
                    print(f"{trade['symbol']}: Exit {trade['reason']} | "
                          f"Entry: {trade['entry_price']:.5f} | "
                          f"Exit: {trade['exit_price']:.5f} | "
                          f"PNL: ${trade['pnl']:,.2f} | "
                          f"Lot: {trade['lot_size']:.2f} | "
                          f"Confidence: {trade['confidence']:.2f}")
        
        return self._get_observation(self.current_step), reward, done, {
            "balance": self.balance,
            "max_drawdown": self.max_drawdown
        }

    def _get_observation(self, step):
        row = self.all_data.iloc[step]
        feats = self.scaler.transform([row.drop(["time", "symbol", "date"]).values])[0]
        oh = np.zeros(len(SYMBOLS))
        oh[self.symbol_to_id[row["symbol"]]] = 1
        return np.concatenate([feats, oh])

# ========== TRAINING ==========
def train_agent():
    env = ForexMultiEnv(ALL_DATA)
    nb_actions = env.action_space.shape[0]
    
    actor = build_actor((1,) + env.observation_space.shape, env.action_space)
    critic = build_critic((1,) + env.observation_space.shape, env.action_space)
    
    memory = SequentialMemory(limit=MEMORY_LIMIT, window_length=1)
    rnd = OrnsteinUhlenbeckProcess(size=nb_actions, theta=0.15, mu=0.0, sigma=0.2, sigma_min=0.05)
    
    agent = DDPGAgent(
        nb_actions=nb_actions,
        actor=actor,
        critic=critic,
        critic_action_input=critic.input[1],
        memory=memory,
        nb_steps_warmup_actor=1000,
        nb_steps_warmup_critic=1000,
        random_process=rnd,
        gamma=DISCOUNT_FACTOR,
        target_model_update=1e-3,
        batch_size=BATCH_SIZE
    )
    
    agent.compile(Adam(learning_rate=1e-4), metrics=["mae"])
    
    latest = get_latest_checkpoint(CHECKPOINT_DIR)
    if latest:
        print(f"[RESUMING] Loading weights from: {latest}")
        agent.load_weights(latest)
    
    print("\nStarting training...")
    callbacks = [CheckpointSaver(CHECKPOINT_DIR, interval=50000)]
    agent.fit(env, nb_steps=TRAIN_STEPS, visualize=False, verbose=0, callbacks=callbacks)
    
    # Save models and stats
    print("\n[SAVING MODELS] Before evaluation...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    actor.save(os.path.join(MODEL_DIR, "actor.h5"))
    agent.save_weights(os.path.join(MODEL_DIR, "ddpg_weights.h5f"), overwrite=True)
    
    stats = {
        "total_episodes": env.current_episode,
        "final_balance": env.balance,
        "total_trades": env.total_trades,
        "max_drawdown": env.max_drawdown
    }
    joblib.dump(stats, os.path.join(MODEL_DIR, "training_stats.pkl"))
    print("[SAVED] All models and stats saved to disk")
    
    # Evaluation
    print("\n=== EVALUATING LEARNED POLICY ===")
    global FIXED_CONFIDENCE_THRESHOLD
    FIXED_CONFIDENCE_THRESHOLD = 0.5
    eval_env = ForexMultiEnv(ALL_DATA)
    eval_env.reset()
    agent.test(eval_env, nb_episodes=5, visualize=False)

if __name__ == "__main__":
    train_agent()
