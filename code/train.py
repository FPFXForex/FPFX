#!/usr/bin/env python3

"""
OPTIMIZED FOREX AI TRAINER - Final Fixed Version
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
from collections import deque, namedtuple, defaultdict
import gym
from gym import spaces
import warnings
from datetime import datetime, timedelta

# Suppress Gym warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gym")

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
    FEATURE_DIM = 26  # 24 time steps x 26 features
    TEMPORAL_DIM = 256
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
    DATA_DIR = os.path.join(BASE_DIR, "Data", "Processed")  # root/FPFX/Data/processed
    NEWS_PATH = os.path.join(BASE_DIR, "Data", "news_cache.csv")  # root/FPFX/Data/news_cache.csv
    SAVE_INTERVAL = 100000
    USE_AMP = True if torch.cuda.is_available() else False
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    PROGRESS_INTERVAL = 120  # Seconds between progress reports

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

# ================ TRADE TRACKING ================
class TradeTracker:
    def __init__(self):
        self.closed_trades = deque(maxlen=10)
        self.open_trades = {}
        self.total_trades = 0
        self.total_pnl = 0
        self.max_balance = Config.INITIAL_BALANCE
        self.highest_balance = Config.INITIAL_BALANCE
        self.start_time = time.time()
        self.last_report_time = self.start_time
        self.period_pnl = 0
        self.period_trades = 0
        
    def open_trade(self, symbol, direction, entry_price, stop_loss, take_profit, lot_size, step_time):
        trade_id = f"{symbol}-{time.time()}"
        self.open_trades[trade_id] = {
            'symbol': symbol,
            'direction': direction,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'lot_size': lot_size,
            'open_time': step_time,
            'open_step': self.total_trades,
            'status': 'open'
        }
        self.total_trades += 1
        return trade_id
        
    def close_trade(self, trade_id, exit_price, exit_time, exit_reason, pnl):
        if trade_id in self.open_trades:
            trade = self.open_trades.pop(trade_id)
            trade.update({
                'exit_price': exit_price,
                'exit_time': exit_time,
                'exit_reason': exit_reason,
                'pnl': pnl,
                'status': 'closed',
                'duration': (exit_time - trade['open_time']).total_seconds() / 60  # in minutes
            })
            self.closed_trades.append(trade)
            self.total_pnl += pnl
            self.period_pnl += pnl
            self.period_trades += 1
            return trade
        return None
        
    def update_max_drawdown(self, current_balance):
        self.max_balance = max(self.max_balance, current_balance)
        drawdown = (self.max_balance - current_balance) / self.max_balance
        return drawdown
        
    def should_report(self):
        current_time = time.time()
        if current_time - self.last_report_time >= Config.PROGRESS_INTERVAL:
            self.last_report_time = current_time
            return True
        return False
        
    def generate_report(self, current_balance, current_step, total_steps):
        elapsed = time.time() - self.start_time
        progress = current_step / total_steps
        drawdown = self.update_max_drawdown(current_balance)
        
        report = f"\n=== Training Progress Report ===\n"
        report += f"Time: {timedelta(seconds=int(elapsed))} | "
        report += f"Progress: {progress:.1%} | "
        report += f"Balance: ${current_balance:,.2f}\n"
        report += f"Max Drawdown: {drawdown:.2%} | "
        report += f"Total PnL: ${self.total_pnl:,.2f} | "
        report += f"Period PnL: ${self.period_pnl:,.2f}\n"
        report += f"Total Trades: {self.total_trades} | "
        report += f"Period Trades: {self.period_trades}\n"
        
        # Reset period stats
        self.period_pnl = 0
        self.period_trades = 0
        
        # Add open trades
        if self.open_trades:
            report += "\n=== Open Trades ===\n"
            for trade_id, trade in list(self.open_trades.items())[-3:]:  # Show last 3 open trades
                report += (f"{trade['symbol']} {'LONG' if trade['direction'] == 1 else 'SHORT'} | "
                          f"Entry: {trade['entry_price']:.5f} | "
                          f"SL: {trade['stop_loss']:.5f} | "
                          f"TP: {trade['take_profit']:.5f} | "
                          f"Size: {trade['lot_size']:.2f} lots\n")
        
        # Add closed trades
        if self.closed_trades:
            report += "\n=== Recent Closed Trades ===\n"
            for trade in list(self.closed_trades)[-3:]:  # Show last 3 closed trades
                report += (f"{trade['symbol']} {'LONG' if trade['direction'] == 1 else 'SHORT'} | "
                          f"P/L: ${trade['pnl']:,.2f} ({'WIN' if trade['pnl'] > 0 else 'LOSS'}) | "
                          f"Entry: {trade['entry_price']:.5f} | "
                          f"Exit: {trade['exit_price']:.5f} | "
                          f"Reason: {trade['exit_reason']}\n")
        
        return report

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
                df['volatility'] = ((df['high'] - df['low']) / 
                                  df['close'].shift(1).replace(0, 1)).fillna(0).astype(np.float32)
                data[symbol] = df
                logger.info("Loaded %d rows for %s", len(df), symbol)
            except Exception as e:
                logger.error("Error loading %s: %s", symbol, str(e))
                raise

        # Load news data
        logger.info("Loading news from: %s", Config.NEWS_PATH)
        news_df = self.load_news_data()

        # Convert date columns to same type before merge
        if not news_df.empty:
            news_df['date'] = pd.to_datetime(news_df['date']).dt.date
            for symbol, df in data.items():
                df['date'] = df['time'].dt.date
                if not news_df.empty:
                    df = pd.merge(df, news_df, how='left', on=['date', 'symbol'])
                    df['news_count'] = df['news_count'].fillna(0).astype(np.float32)
                    df['avg_sentiment'] = df['avg_sentiment'].fillna(0).astype(np.float32)
                data[symbol] = df
        return data

    def load_news_data(self):
        try:
            if os.path.exists(Config.NEWS_PATH):
                news_df = pd.read_csv(Config.NEWS_PATH)
                if 'date' in news_df.columns:
                    news_df['date'] = pd.to_datetime(news_df['date']).dt.date
                return news_df
            logger.warning("News file not found at: %s", Config.NEWS_PATH)
            return pd.DataFrame(columns=['date', 'symbol', 'news_count', 'avg_sentiment'])
        except Exception as e:
            logger.error("Error loading news: %s", str(e))
            return pd.DataFrame(columns=['date', 'symbol', 'news_count', 'avg_sentiment'])

    def create_scalers(self):
        scalers = {}
        all_data = pd.concat(self.data.values())

        tech_cols = ['RSI_14', 'BB_%B', 'ATR_14', 'STOCH_%K', 'STOCH_%D',
                    'MACD_line', 'MACD_signal', 'KC_upper', 'KC_middle', 'KC_lower',
                    'SMA_50', 'ADX_14', 'PSAR', 'SMA_200', 'TRIX_15',
                    'Regime0', 'Regime1', 'Regime2', 'Regime3', 'volatility']
        scalers['tech'] = RobustScaler().fit(all_data[tech_cols])

        price_cols = ['open', 'high', 'low', 'close']
        scalers['price'] = RobustScaler().fit(all_data[price_cols])

        return scalers

    def get_sequence(self, symbol, index, seq_len):
        df = self.data[symbol]
        if index < seq_len or index >= len(df) - 4:
            return None

        seq = df.iloc[index-seq_len:index]

        tech = self.scalers['tech'].transform(seq[[
            'RSI_14', 'BB_%B', 'ATR_14', 'STOCH_%K', 'STOCH_%D',
            'MACD_line', 'MACD_signal', 'KC_upper', 'KC_middle', 'KC_lower',
            'SMA_50', 'ADX_14', 'PSAR', 'SMA_200', 'TRIX_15',
            'Regime0', 'Regime1', 'Regime2', 'Regime3', 'volatility'
        ]])

        price = self.scalers['price'].transform(seq[['open', 'high', 'low', 'close']])
        temporal = np.stack([seq['hour'].values, seq['day_of_week'].values], axis=-1)

        features = np.concatenate([tech, price, temporal], axis=1)
        return {
            'features': features.astype(np.float32),
            'current_price': df.iloc[index]['close'],
            'symbol': symbol,
            'time': df.iloc[index]['time']
        }

# ================ MODEL ARCHITECTURE ================
class TemporalFeatureExtractor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, Config.TEMPORAL_DIM, batch_first=True)
        self.layer_norm = nn.LayerNorm(Config.TEMPORAL_DIM)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = x.float()  # Ensure float32
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, TEMPORAL_DIM)
        lstm_out = self.layer_norm(lstm_out)
        return lstm_out[:, -1, :]  # (batch_size, TEMPORAL_DIM)

class ForexPolicyNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.feature_extractor = TemporalFeatureExtractor(input_dim)
        
        # Policy head
        self.policy_fc = nn.Sequential(
            nn.Linear(Config.TEMPORAL_DIM, Config.POLICY_DIM),
            nn.ReLU(),
            nn.Linear(Config.POLICY_DIM, 10)  # 5 means + 5 stds
        )
        
        # Value head
        self.value_fc = nn.Sequential(
            nn.Linear(Config.TEMPORAL_DIM, Config.POLICY_DIM),
            nn.ReLU(),
            nn.Linear(Config.POLICY_DIM, 1)
        )
        
        self.action_bounds = torch.tensor([
            [0.0, 1.0], [1.0, 4.0], [1.5, 6.0], [-0.3, 0.3], [0.0, 1.0]
        ], device=Config.DEVICE)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
            
        features = self.feature_extractor(x)
        policy_params = self.policy_fc(features)
        
        means = policy_params[..., :5]
        stds = torch.nn.functional.softplus(policy_params[..., 5:]) + 1e-6  # Add small epsilon
        
        # Clip means to reasonable bounds
        means = torch.sigmoid(means) * 2 - 1  # Scale to [-1, 1] range
        
        value = self.value_fc(features)
        
        return means.float(), stds.float(), value.float()

    def act(self, state):
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state).to(Config.DEVICE)
            if state.dim() == 2:
                state = state.unsqueeze(0)
                
            means, stds, value = self.forward(state)
            
            # Create distribution with numerical stability checks
            stds = torch.clamp(stds, min=1e-6, max=1.0)
            dist = Normal(means, stds)
            
            action = dist.sample()
            action = torch.tanh(action)  # Bound actions to [-1, 1]
            
            low, high = self.action_bounds[:, 0], self.action_bounds[:, 1]
            scaled_action = low + (0.5 * (action + 1.0)) * (high - low)
            
            return scaled_action.squeeze().cpu().numpy(), dist.log_prob(action).sum(-1).item(), value.item()

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
            low=-10, high=10, shape=(24, Config.FEATURE_DIM), dtype=np.float32
        )
        self.trade_tracker = TradeTracker()
        self.reset()

    def reset(self):
        self.balance = Config.INITIAL_BALANCE
        self.open_positions = {}
        self.current_symbol = np.random.choice(self.symbols)
        return self._get_observation(self.current_symbol)

    def _get_observation(self, symbol):
        seq_data = self.data_engine.get_sequence(symbol, self.current_step[symbol], 24)
        if seq_data is None:
            return np.zeros((24, Config.FEATURE_DIM), dtype=np.float32)
        return seq_data['features']

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
        step_time = row["time"]
        
        # Ensure action has 5 elements
        if len(action) != 5:
            logger.error(f"Invalid action shape: {len(action)}, expected 5 elements")
            action = np.array([0.5, 2.5, 3.0, 0.0, 0.5])  # Default safe action
        
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
            
            trade_id = self.trade_tracker.open_trade(
                symbol, direction, entry_price, stop_loss, take_profit, lot_size, step_time
            )
            
            self.open_positions[trade_id] = {
                'direction': direction,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'lot_size': lot_size,
                'open_step': self.current_step[symbol]
            }
            trade_opened = True

        reward = 0
        for trade_id, pos in list(self.open_positions.items()):
            if pos.get('symbol', symbol) != symbol:
                continue

            direction = pos['direction']
            exit_price = None
            exit_reason = None
            penalty = False

            # SL exit
            if (direction == 1 and low <= pos['stop_loss']) or (direction == -1 and high >= pos['stop_loss']):
                exit_price = pos['stop_loss']
                exit_reason = "SL"
                penalty = True
            # TP exit
            elif (direction == 1 and high >= pos['take_profit']) or (direction == -1 and low <= pos['take_profit']):
                exit_price = pos['take_profit']
                exit_reason = "TP"
            # Time exit
            elif (self.current_step[symbol] - pos['open_step']) >= Config.MAX_HOLD_BARS:
                exit_price = current_price
                exit_reason = "TIME"

            if exit_price is not None:
                price_diff = exit_price - pos['entry_price']
                if "JPY" in symbol:
                    pnl = price_diff * direction * pos['lot_size'] * 1000 / exit_price
                else:
                    pnl = price_diff * direction * pos['lot_size'] * 100000
                
                pnl -= Config.COMMISSION * pos['lot_size']
                self.balance += pnl
                
                if penalty:
                    reward += pnl - abs(pnl) * 0.3
                else:
                    reward += pnl + abs(pnl) * 0.2
                
                # Track the closed trade
                self.trade_tracker.close_trade(
                    trade_id, exit_price, step_time, exit_reason, pnl
                )
                del self.open_positions[trade_id]

        if not trade_opened and row['volatility'] > 0.002:
            reward -= 10.0
        
        reward -= 0.05
        self.current_step[symbol] += 1
        self.current_symbol = np.random.choice(self.symbols)
        
        done = (self.balance < Config.INITIAL_BALANCE * 0.75 or
                self.current_step[symbol] >= len(self.data_engine.data[symbol]) - 10)
        
        # Generate progress report if needed
        if self.trade_tracker.should_report():
            logger.info(self.trade_tracker.generate_report(self.balance, self.current_step[symbol], Config.TIMESTEPS))
        
        # Convert done to Python bool to avoid numpy.bool_ warning
        done = bool(done)
        
        return self._get_observation(self.current_symbol), reward, done, {
            "balance": self.balance,
            "current_price": current_price,
            "symbol": symbol
        }

    def update_confidence(self, progress):
        if progress < Config.EXPLORATION_DECAY:
            self.confidence_threshold = Config.INIT_CONFIDENCE + (Config.FINAL_CONFIDENCE - Config.INIT_CONFIDENCE) * (progress / Config.EXPLORATION_DECAY)
        else:
            self.confidence_threshold = Config.FINAL_CONFIDENCE

# ================ PPO TRAINING ================
class ForexPPOTrainer:
    def __init__(self, env):
        self.env = env
        self.policy = ForexPolicyNetwork(Config.FEATURE_DIM).to(Config.DEVICE)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=Config.LR)
        self.scaler = torch.cuda.amp.GradScaler(enabled=Config.USE_AMP)
        self.writer = SummaryWriter(Config.LOG_DIR)
        self.Transition = namedtuple('Transition', ['state', 'action', 'log_prob', 'value', 'reward', 'done'])
        self.memory = deque(maxlen=Config.BATCH_SIZE)

    def train(self):
        state = self.env.reset()
        state_tensor = torch.FloatTensor(state).to(Config.DEVICE).unsqueeze(0)
        global_step = 0

        while global_step < Config.TIMESTEPS:
            progress = global_step / Config.TIMESTEPS
            self.env.update_confidence(progress)

            action, log_prob, value = self.policy.act(state_tensor)
            next_state, reward, done, info = self.env.step(action)
            next_state_tensor = torch.FloatTensor(next_state).to(Config.DEVICE).unsqueeze(0)

            # Convert done to Python bool before creating tensor
            done_bool = bool(done)
            self.memory.append(self.Transition(
                state_tensor.float(),
                torch.FloatTensor(action).to(Config.DEVICE),
                torch.FloatTensor([log_prob]).to(Config.DEVICE),
                torch.FloatTensor([value]).to(Config.DEVICE),
                torch.FloatTensor([reward]).to(Config.DEVICE),
                torch.FloatTensor([done_bool]).to(Config.DEVICE)  # Use Python bool here
            ))

            state_tensor = next_state_tensor if not done else torch.FloatTensor(
                self.env.reset()).to(Config.DEVICE).unsqueeze(0)
            
            global_step += 1

            if len(self.memory) >= Config.BATCH_SIZE:
                self.update_model()

            if global_step % Config.SAVE_INTERVAL == 0:
                self.save_checkpoint(global_step, info['balance'])

        self.save_checkpoint(global_step, info['balance'], final=True)
        logger.info("Training complete")

    def update_model(self):
        states = torch.cat([t.state for t in self.memory]).float()
        actions = torch.stack([t.action for t in self.memory]).float()
        old_log_probs = torch.stack([t.log_prob for t in self.memory]).float()
        old_values = torch.stack([t.value for t in self.memory]).float()
        rewards = torch.stack([t.reward for t in self.memory]).float()
        dones = torch.stack([t.done for t in self.memory]).float()
        self.memory.clear()

        returns = torch.zeros_like(rewards)
        with torch.no_grad():
            _, _, last_value = self.policy(states[-1:])
            returns[-1] = rewards[-1] + Config.GAMMA * (1 - dones[-1]) * last_value.squeeze()

        for t in reversed(range(len(returns)-1)):
            returns[t] = rewards[t] + Config.GAMMA * (1 - dones[t]) * returns[t+1]

        advantages = returns - old_values.squeeze()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(Config.N_EPOCHS):
            with torch.cuda.amp.autocast(enabled=Config.USE_AMP):
                means, stds, values = self.policy(states)
                
                # Numerical stability checks
                means = means.float()
                stds = torch.clamp(stds.float(), min=1e-6, max=1.0)
                
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
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

    # Training summary
    duration = time.time() - start_time
    logger.info("Training completed in %.2f hours", duration/3600)
    logger.info("Models saved to: %s", Config.MODEL_DIR)
