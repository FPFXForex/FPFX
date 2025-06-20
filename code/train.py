#!/usr/bin/env python3
"""
OPTIMIZED FOREX AI TRAINER - Correct Paths for FPFX Structure
"""
import os
import gc
import math
import time
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal
from sklearn.preprocessing import RobustScaler
from collections import deque, namedtuple
import gym
from gym import spaces

# ================ CONFIG WITH FPFX PATHS ================
class Config:
    # Core parameters
    SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "XAUUSD"]
    INITIAL_BALANCE = 100_000.0
    TIMESTEPS = 1_200_000
    BATCH_SIZE = 8192
    GAMMA = 0.99
    
    # Risk management
    MAX_OPEN_TRADES = 3
    DAILY_DD_LIMIT = 0.20
    MIN_RISK = 0.005
    MAX_RISK = 0.025
    COMMISSION = 0.0002
    MIN_LOT = 0.1
    MAX_LOT = 15.0
    MIN_STOP_DISTANCE_PIPS = 10
    MAX_ATR_PIPS = 100
    MAX_HOLD_BARS = 24
    
    # Exploration strategy
    INIT_CONFIDENCE = 0.1
    FINAL_CONFIDENCE = 0.82
    EXPLORATION_DECAY = 0.65
    
    # Neural network architecture
    FEATURE_DIM = 64
    TEMPORAL_DIM = 128
    POLICY_DIM = 256
    
    # Training optimization
    LR = 2.5e-4
    CLIP_PARAM = 0.15
    ENTROPY_COEF = 0.02
    VALUE_COEF = 0.6
    GRAD_CLIP = 0.8
    N_EPOCHS = 3
    
    # System - FPFX Directory Structure
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # root/FPFX
    MODEL_DIR = os.path.join(BASE_DIR, "model")  # root/FPFX/model
    LOG_DIR = os.path.join(BASE_DIR, "logs")
    DATA_DIR = os.path.join(BASE_DIR, "Data", "processed")  # root/FPFX/Data/processed
    NEWS_PATH = os.path.join(BASE_DIR, "Data", "news_cache.csv")  # root/FPFX/Data/news_cache.csv
    SAVE_INTERVAL = 100000
    USE_AMP = True if torch.cuda.is_available() else False
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    @classmethod
    def setup_directories(cls):
        os.makedirs(cls.MODEL_DIR, exist_ok=True)
        os.makedirs(cls.LOG_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(cls.NEWS_PATH), exist_ok=True)

# Initialize directories
Config.setup_directories()
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ForexTrainer")
logger.info("Using device: %s", Config.DEVICE)
logger.info("Model directory: %s", Config.MODEL_DIR)
logger.info("Data directory: %s", Config.DATA_DIR)

# ================ DATA ENGINEERING ================
class ForexDataEngine:
    def __init__(self):
        self.data = self.load_and_process()
        self.scalers = self.create_scalers()
        
    def load_and_process(self):
        logger.info("Loading data from: %s", Config.DATA_DIR)
        data = {}
        
        for symbol in Config.SYMBOLS:
            path = os.path.join(Config.DATA_DIR, f"{symbol}_processed.csv")
            logger.info("Processing: %s", path)
            try:
                df = pd.read_csv(path, parse_dates=["time"])
                df["symbol"] = symbol
                
                # Essential features
                df['hour'] = df['time'].dt.hour.astype(np.float32)
                df['day_of_week'] = df['time'].dt.dayofweek.astype(np.float32)
                df['volatility'] = ((df['high'] - df['low']) / df['close'].shift(1).replace(0, 1)).fillna(0).astype(np.float32)
                
                data[symbol] = df
                logger.info("Loaded %d rows for %s", len(df), symbol)
            except Exception as e:
                logger.error("Error loading %s: %s", symbol, str(e))
                raise
        
        # Load news data
        logger.info("Loading news from: %s", Config.NEWS_PATH)
        news_df = self.load_news_data()
        for symbol, df in data.items():
            df['date'] = df['time'].dt.date
            df = pd.merge(df, news_df, how='left', on=['date', 'symbol'])
            df['news_count'] = df['news_count'].fillna(0).astype(np.float32)
            df['avg_sentiment'] = df['avg_sentiment'].fillna(0).astype(np.float32)
            data[symbol] = df
        
        return data
    
    def load_news_data(self):
        try:
            if os.path.exists(Config.NEWS_PATH):
                return pd.read_csv(Config.NEWS_PATH, parse_dates=['date'])
            logger.warning("News file not found at: %s", Config.NEWS_PATH)
            return pd.DataFrame(columns=['date', 'symbol', 'news_count', 'avg_sentiment'])
        except Exception as e:
            logger.error("Error loading news: %s", str(e))
            return pd.DataFrame(columns=['date', 'symbol', 'news_count', 'avg_sentiment'])
    
    def create_scalers(self):
        scalers = {}
        all_data = pd.concat(self.data.values())
        
        tech_cols = ['RSI_14', 'BB_%B', 'ATR_14', 'STOCH_%K', 'STOCH_%D', 'volatility']
        scalers['tech'] = RobustScaler().fit(all_data[tech_cols])
        
        price_cols = ['open', 'high', 'low', 'close']
        scalers['price'] = RobustScaler().fit(all_data[price_cols])
        
        return scalers
    
    def get_sequence(self, symbol, index, seq_len):
        df = self.data[symbol]
        if index < seq_len or index >= len(df) - 4:
            return None
            
        seq = df.iloc[index-seq_len:index]
        
        tech = self.scalers['tech'].transform(seq[['RSI_14', 'BB_%B', 'ATR_14', 'STOCH_%K', 'STOCH_%D', 'volatility']])
        price = self.scalers['price'].transform(seq[['open', 'high', 'low', 'close']])
        temporal = np.stack([seq['hour'].values, seq['day_of_week'].values], axis=-1)
        
        features = np.concatenate([tech, price, temporal], axis=1)
        
        return {
            'features': features.astype(np.float32),
            'current_price': df.iloc[index]['close'],
            'symbol': symbol
        }

# ================ MODEL ARCHITECTURE ================
class TemporalFeatureExtractor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, Config.TEMPORAL_DIM//2, 
                           batch_first=True, bidirectional=True)
        self.conv = nn.Conv1d(input_dim, Config.TEMPORAL_DIM//2, kernel_size=3, padding=1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        cnn_out = x.transpose(1, 2)
        cnn_out = torch.relu(self.conv(cnn_out)).transpose(1, 2)
        return torch.cat([lstm_out, cnn_out], dim=-1)[:, -1, :]

class ForexPolicyNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.feature_extractor = TemporalFeatureExtractor(input_dim)
        
        self.policy_fc = nn.Sequential(
            nn.Linear(Config.TEMPORAL_DIM, Config.POLICY_DIM),
            nn.ReLU(),
            nn.Linear(Config.POLICY_DIM, 10)
        )
        
        self.value_fc = nn.Sequential(
            nn.Linear(Config.TEMPORAL_DIM, Config.POLICY_DIM),
            nn.ReLU(),
            nn.Linear(Config.POLICY_DIM, 1)
        )
        
        self.action_bounds = torch.tensor([
            [0.0, 1.0], [1.0, 4.0], [1.5, 6.0], [-0.3, 0.3], [0.0, 1.0]
        ], device=Config.DEVICE)
        
    def forward(self, x):
        features = self.feature_extractor(x)
        policy_params = self.policy_fc(features)
        means = policy_params[..., :5]
        stds = torch.nn.functional.softplus(policy_params[..., 5:]) + 1e-5
        value = self.value_fc(features)
        return means, stds, value
    
    def act(self, state):
        with torch.no_grad():
            means, stds, value = self.forward(state)
            dist = Normal(means, stds)
            action = dist.sample()
            
            action = torch.tanh(action)
            low, high = self.action_bounds[:, 0], self.action_bounds[:, 1]
            scaled_action = low + (0.5 * (action + 1.0)) * (high - low)
            
            return scaled_action.cpu().numpy(), dist.log_prob(action).sum(-1).cpu().numpy(), value.cpu().numpy()

# ================ TRADING ENVIRONMENT ================
class ForexTradingEnv(gym.Env):
    def __init__(self, data_engine):
        super().__init__()
        self.data_engine = data_engine
        self.symbols = Config.SYMBOLS
        self.current_step = {s: 24 for s in self.symbols}
        self.confidence_threshold = Config.INIT_CONFIDENCE
        self.pip_sizes = {
            "EURUSD": 0.0001, "GBPUSD": 0.0001, "USDJPY": 0.01,
            "AUDUSD": 0.0001, "USDCAD": 0.0001, "XAUUSD": 0.01
        }
        
        self.action_space = spaces.Box(
            low=np.array([0.0, 1.0, 1.5, -0.3, 0.0]),
            high=np.array([1.0, 4.0, 6.0, 0.3, 1.0]),
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=-10, high=10, shape=(24, 12), dtype=np.float32
        )
        
        self.reset()
    
    def reset(self):
        self.balance = Config.INITIAL_BALANCE
        self.open_positions = {}
        self.current_symbol = np.random.choice(self.symbols)
        return self._get_observation(self.current_symbol)
    
    def _get_observation(self, symbol):
        seq_data = self.data_engine.get_sequence(symbol, self.current_step[symbol], 24)
        return seq_data['features'] if seq_data else np.zeros((24, 12), dtype=np.float32)
    
    def _calculate_stops(self, entry_price, direction, atr, sl_mult, tp_mult, symbol):
        scaled_atr = atr * 10
        pip_size = self.pip_sizes[symbol]
        atr_pips = min(max(scaled_atr / pip_size, 10), 100)
        
        sl_pips = atr_pips * sl_mult
        tp_pips = atr_pips * tp_mult
        
        if tp_pips / sl_pips < 1.3:
            tp_pips = sl_pips * 1.3
            
        if direction == 1:
            return entry_price - (sl_pips * pip_size), entry_price + (tp_pips * pip_size)
        else:
            return entry_price + (sl_pips * pip_size), entry_price - (tp_pips * pip_size)
    
    def step(self, action):
        symbol = self.current_symbol
        row = self.data_engine.data[symbol].iloc[self.current_step[symbol]]
        current_price = row["close"]
        high = row["high"]
        low = row["low"]
        atr = row["ATR_14"]
        
        signal, sl_mult, tp_mult, sentiment_exit, confidence = action
        
        trade_opened = False
        if (confidence >= self.confidence_threshold and 
            symbol not in self.open_positions and
            len(self.open_positions) < Config.MAX_OPEN_TRADES):
            
            direction = 1 if signal >= 0.5 else -1
            entry_price = high * 1.0001 if direction == 1 else low * 0.9999
            stop_loss, take_profit = self._calculate_stops(
                entry_price, direction, atr, sl_mult, tp_mult, symbol
            )
            
            pip_size = self.pip_sizes[symbol]
            stop_distance = abs(entry_price - stop_loss) / pip_size
            pip_value = 10.0 if symbol != "USDJPY" else 1000 / entry_price
            risk_amount = Config.MIN_RISK + confidence * (Config.MAX_RISK - Config.MIN_RISK)
            lot_size = min(max((risk_amount * self.balance) / (stop_distance * pip_value), 0.1), 15.0)
            
            self.open_positions[symbol] = {
                'direction': direction,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'lot_size': lot_size,
                'open_step': self.current_step[symbol]
            }
            trade_opened = True
        
        reward = 0
        for sym, pos in list(self.open_positions.items()):
            if sym != symbol:
                continue
                
            direction = pos['direction']
            exit_price = None
            penalty = False
            
            # SL exit
            if (direction == 1 and low <= pos['stop_loss']) or (direction == -1 and high >= pos['stop_loss']):
                exit_price = pos['stop_loss']
                penalty = True
            # TP exit
            elif (direction == 1 and high >= pos['take_profit']) or (direction == -1 and low <= pos['take_profit']):
                exit_price = pos['take_profit']
            # Time exit
            elif (self.current_step[sym] - pos['open_step']) >= 24:
                exit_price = current_price
            
            if exit_price is not None:
                price_diff = exit_price - pos['entry_price']
                if "JPY" in sym:
                    pnl = price_diff * direction * pos['lot_size'] * 1000 / exit_price
                else:
                    pnl = price_diff * direction * pos['lot_size'] * 100000
                pnl -= Config.COMMISSION * pos['lot_size']
                
                if penalty:
                    reward += pnl - abs(pnl) * 0.3
                else:
                    reward += pnl + abs(pnl) * 0.2
                
                del self.open_positions[sym]
        
        if not trade_opened and row['volatility'] > 0.002:
            reward -= 10.0
        
        reward -= 0.05
        
        self.current_step[symbol] += 1
        self.current_symbol = np.random.choice(self.symbols)
        
        done = (self.balance < Config.INITIAL_BALANCE * 0.75 or 
                self.current_step[symbol] >= len(self.data_engine.data[symbol]) - 10)
        
        return self._get_observation(self.current_symbol), reward, done, {"balance": self.balance}
    
    def update_confidence(self, progress):
        if progress < Config.EXPLORATION_DECAY:
            self.confidence_threshold = Config.INIT_CONFIDENCE + (Config.FINAL_CONFIDENCE - Config.INIT_CONFIDENCE) * (progress / Config.EXPLORATION_DECAY)
        else:
            self.confidence_threshold = Config.FINAL_CONFIDENCE

# ================ PPO TRAINING ================
class ForexPPOTrainer:
    def __init__(self, env):
        self.env = env
        self.policy = ForexPolicyNetwork(
            self.env.observation_space.shape[-1]
        ).to(Config.DEVICE)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=Config.LR)
        self.scaler = torch.cuda.amp.GradScaler(enabled=Config.USE_AMP)
        self.writer = SummaryWriter(Config.LOG_DIR)
        
        self.Transition = namedtuple('Transition', ['state', 'action', 'log_prob', 'value', 'reward', 'done'])
        self.memory = deque(maxlen=Config.BATCH_SIZE)
    
    def train(self):
        state = self.env.reset()
        state_tensor = torch.tensor(state, dtype=torch.float32, device=Config.DEVICE).unsqueeze(0)
        global_step = 0
        
        while global_step < Config.TIMESTEPS:
            progress = global_step / Config.TIMESTEPS
            self.env.update_confidence(progress)
            
            action, log_prob, value = self.policy.act(state_tensor)
            next_state, reward, done, info = self.env.step(action)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32, device=Config.DEVICE).unsqueeze(0)
            
            self.memory.append(self.Transition(
                state_tensor, 
                torch.tensor(action, dtype=torch.float32),
                torch.tensor(log_prob, dtype=torch.float32),
                torch.tensor(value, dtype=torch.float32),
                torch.tensor(reward, dtype=torch.float32),
                torch.tensor(done, dtype=torch.float32)
            ))
            
            state_tensor = next_state_tensor if not done else torch.tensor(
                self.env.reset(), dtype=torch.float32, device=Config.DEVICE).unsqueeze(0)
            
            global_step += 1
            
            if len(self.memory) >= Config.BATCH_SIZE:
                self.update_model()
            
            if global_step % Config.SAVE_INTERVAL == 0:
                self.save_checkpoint(global_step, info['balance'])
        
        self.save_checkpoint(global_step, info['balance'], final=True)
        logger.info("Training complete")
    
    def update_model(self):
        states = torch.cat([t.state for t in self.memory])
        actions = torch.stack([t.action for t in self.memory])
        old_log_probs = torch.stack([t.log_prob for t in self.memory])
        old_values = torch.stack([t.value for t in self.memory])
        rewards = torch.stack([t.reward for t in self.memory])
        dones = torch.stack([t.done for t in self.memory])
        self.memory.clear()
        
        returns = torch.zeros_like(rewards)
        last_value = self.policy(states[-1:])[2]
        returns[-1] = rewards[-1] + Config.GAMMA * (1 - dones[-1]) * last_value
        for t in reversed(range(len(returns)-1)):
            returns[t] = rewards[t] + Config.GAMMA * (1 - dones[t]) * returns[t+1]
        
        advantages = returns - old_values.squeeze()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(Config.N_EPOCHS):
            with torch.cuda.amp.autocast(enabled=Config.USE_AMP):
                means, stds, values = self.policy(states)
                dist = Normal(means, stds)
                log_probs = dist.log_prob(actions).sum(-1)
                
                ratio = (log_probs - old_log_probs).exp()
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1-Config.CLIP_PARAM, 1+Config.CLIP_PARAM) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = 0.5 * (returns - values.squeeze()).pow(2).mean()
                
                entropy = dist.entropy().mean()
                
                loss = policy_loss + Config.VALUE_COEF * value_loss - Config.ENTROPY_COEF * entropy
            
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), Config.GRAD_CLIP)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        
        self.writer.add_scalar("Loss/Total", loss.item(), global_step)
    
    def save_checkpoint(self, step, balance, final=False):
        path = os.path.join(Config.MODEL_DIR, f"forex_{'final' if final else step}.pth")
        torch.save({
            'step': step,
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'balance': balance
        }, path)
        logger.info("Saved checkpoint at step %d, balance $%.2f", step, balance)

# ================ MAIN EXECUTION ================
if __name__ == "__main__":
    logger.info("Starting FPFX Forex AI Training")
    logger.info("Base directory: %s", Config.BASE_DIR)
    
    # Estimate training time
    start_time = time.time()
    
    # Initialize and train
    data_engine = ForexDataEngine()
    env = ForexTradingEnv(data_engine)
    trainer = ForexPPOTrainer(env)
    trainer.train()
    
    # Training summary
    duration = time.time() - start_time
    logger.info("Training completed in %.2f hours", duration/3600)
    logger.info("Models saved to: %s", Config.MODEL_DIR)
