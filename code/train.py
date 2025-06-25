#!/usr/bin/env python3

"""

FOREX AI TRAINER - Enhanced with Full Indicator Utilization

"""

import os
import gc
import time
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import StandardScaler
from collections import deque, namedtuple
import gym
from gym import spaces
import warnings
from datetime import datetime
import random
import math

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("forex_ai_trainer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ForexTrainer")

# ================ CONFIGURATION ================
class Config:
    SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "XAUUSD"]
    INITIAL_BALANCE = 100_000.0
    TIMESTEPS = 1_000_000
    BATCH_SIZE = 1024  # Reduced from 4096 to 1024
    GRAD_ACCUM_STEPS = 4  # New: gradient accumulation steps
    GAMMA = 0.99
    MAX_OPEN_TRADES = 3
    DAILY_DD_LIMIT = 0.20
    MIN_RISK = 0.005
    MAX_RISK = 0.015  # 1.5% max risk
    COMMISSION = 0.0002
    MIN_LOT = 0.01
    MAX_LOT = 100.0
    MIN_STOP_DISTANCE_PIPS = 15
    MAX_ATR_PIPS = 100
    FEATURE_DIM = 31  # FIXED: Updated from 24 to 31 (actual feature count)
    POLICY_DIM = 128  # Reduced from 256 to 128
    LR = 3e-4
    CLIP_PARAM = 0.2
    ENTROPY_COEF = 0.02
    VALUE_COEF = 0.5
    GRAD_CLIP = 0.5
    N_EPOCHS = 4
    SAVE_INTERVAL = 50_000
    SEQ_LEN = 24
    CONFIDENCE_THRESHOLD_START = 0.0
    CONFIDENCE_THRESHOLD_END = 0.5
    THRESHOLD_TRANSITION_START = 0.5
    THRESHOLD_TRANSITION_END = 0.8
    TRADE_HISTORY_SIZE = 6  # Changed from 10 to 6
    PROGRESS_INTERVAL = 30
    XAUUSD_PIP_VALUE = 10.0
    XAUUSD_LOT_MULTIPLIER = 0.01
    PIP_SIZES = {
        "EURUSD": 0.0001, "GBPUSD": 0.0001, "USDJPY": 0.01,
        "AUDUSD": 0.0001, "USDCAD": 0.0001, "XAUUSD": 0.1
    }
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "Data/Processed")
    MODEL_DIR = os.path.join(BASE_DIR, "model")
    LOG_DIR = os.path.join(BASE_DIR, "logs")
    NEWS_PATH = os.path.join(BASE_DIR, "Data/news_cache.csv")

    @classmethod
    def setup_directories(cls):
        os.makedirs(cls.MODEL_DIR, exist_ok=True)
        os.makedirs(cls.LOG_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(cls.NEWS_PATH), exist_ok=True)

Config.setup_directories()
logger.info(f"Using device: {Config.DEVICE}")
logger.info(f"Data directory: {Config.DATA_DIR}")

# ================ DATA ENGINE ================
class ForexDataEngine:
    def __init__(self):
        self.news_df = self.load_news_data()
        self.data = self.load_data()
        self.scalers = self.create_scalers()
        self.feature_cols = self.get_feature_columns()
        logger.info("Data loaded successfully")

    def load_news_data(self):
        try:
            if not os.path.exists(Config.NEWS_PATH):
                logger.warning(f"News file not found: {Config.NEWS_PATH}")
                return pd.DataFrame()
            news_df = pd.read_csv(Config.NEWS_PATH, parse_dates=["date"])
            news_df["date_str"] = news_df["date"].dt.strftime('%Y-%m-%d')
            logger.info(f"Loaded news data: {len(news_df)} records")
            return news_df
        except Exception as e:
            logger.error(f"Error loading news data: {str(e)}")
            return pd.DataFrame()

    def get_feature_columns(self):
        # Consolidated feature list using all available indicators
        return [
            'open', 'high', 'low', 'close',
            'RSI_14', 'BB_%B', 'BB_width', 'ATR_14',
            'STOCH_%K', 'STOCH_%D',
            'MACD_line', 'MACD_signal', 'MACD_hist',
            'volatility', 'momentum',
            'ADX_14', 'DMP_14', 'DMN_14',
            'SMA_50', 'SMA_200', 'TRIX_15',
            'volume_ma', 'volume_pct',
            'Regime0', 'Regime1', 'Regime2', 'Regime3',
            'hour', 'day_of_week',
            'news_count', 'avg_sentiment'
        ]

    def load_data(self):
        data = {}
        for symbol in Config.SYMBOLS:
            try:
                file_path = os.path.join(Config.DATA_DIR, f"{symbol}_processed.csv")
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"Data file not found: {file_path}")
                df = pd.read_csv(file_path, parse_dates=["timestamp"])
                df["symbol"] = symbol
                df["date_str"] = df["timestamp"].dt.strftime('%Y-%m-%d')
                
                # Add XAUUSD-specific features to common structure
                if symbol == "XAUUSD":
                    df["BB_width"] = df["BB_width"].fillna(0)
                    df["KC_upper"] = df["KC_upper"].fillna(0)
                    df["KC_middle"] = df["KC_middle"].fillna(0)
                    df["KC_lower"] = df["KC_lower"].fillna(0)
                    # Create synthetic features for consistency
                    df["BB_%B"] = (df["close"] - df["KC_lower"]) / (df["KC_upper"] - df["KC_lower"]).replace(0, 1)
                    df["BB_width"] = df["KC_upper"] - df["KC_lower"]
                
                # Merge news data
                if not self.news_df.empty:
                    # Drop existing columns to prevent duplicates
                    if 'news_count' in df.columns:
                        df.drop(columns=['news_count'], inplace=True)
                    if 'avg_sentiment' in df.columns:
                        df.drop(columns=['avg_sentiment'], inplace=True)
                    df = pd.merge(
                        df,
                        self.news_df[["date_str", "symbol", "news_count", "avg_sentiment"]],
                        how="left",
                        on=["date_str", "symbol"]
                    )
                    df[["news_count", "avg_sentiment"]] = df[["news_count", "avg_sentiment"]].fillna(0)
                else:
                    df["news_count"] = 0
                    df["avg_sentiment"] = 0
                
                # Process time features
                df['hour'] = df['timestamp'].dt.hour
                df['day_of_week'] = df['timestamp'].dt.dayofweek
                
                # Clean data
                df.replace([np.inf, -np.inf], 0, inplace=True)
                df.fillna(0, inplace=True)
                
                # Add any missing features
                for col in self.get_feature_columns():
                    if col not in df.columns:
                        df[col] = 0
                        logger.warning(f"Added missing column {col} to {symbol} with default 0")
                
                data[symbol] = df
                logger.info(f"Loaded {len(df)} rows for {symbol} with {len(df.columns)} features")
                logger.debug(f"Columns: {list(df.columns)}")
            except Exception as e:
                logger.error(f"Error loading {symbol}: {str(e)}", exc_info=True)
                raise
        return data

    def create_scalers(self):
        all_data = pd.concat(self.data.values())
        numeric_cols = [col for col in self.get_feature_columns()
                        if col not in ['hour', 'day_of_week', 'Regime0', 'Regime1', 'Regime2', 'Regime3']]
        scaler = StandardScaler()
        scaler.fit(all_data[numeric_cols])
        return scaler

    def get_sequence(self, symbol, index):
        df = self.data[symbol]
        if index < Config.SEQ_LEN or index >= len(df):
            return None
        seq = df.iloc[index-Config.SEQ_LEN:index]
        bar = df.iloc[index]
        
        # Prepare features
        numeric_cols = [col for col in self.get_feature_columns()
                        if col not in ['hour', 'day_of_week', 'Regime0', 'Regime1', 'Regime2', 'Regime3']]
        scaled = self.scalers.transform(seq[numeric_cols])
        
        # Add categorical/time features
        time_features = seq[['hour', 'day_of_week']].values
        regime_features = seq[['Regime0', 'Regime1', 'Regime2', 'Regime3']].values
        feats = np.concatenate([scaled, time_features, regime_features], axis=1)
        feats = np.nan_to_num(feats, nan=0.0)
        
        return {
            'features': feats.astype(np.float32),
            'open': bar['open'],
            'high': bar['high'],
            'low': bar['low'],
            'current_price': bar['close'],
            'atr': bar['ATR_14'],
            'news_count': bar['news_count'],
            'avg_sentiment': bar['avg_sentiment']
        }

# ================ TRADING ENVIRONMENT ================
class ForexTradingEnv(gym.Env):
    def __init__(self, data_engine):
        super().__init__()
        self.data_engine = data_engine
        self.symbols = Config.SYMBOLS
        self.pip = Config.PIP_SIZES
        
        # Action space: 6 × [signal, sl_mult, tp_mult, conf]
        self.action_space = spaces.Box(
            low=np.zeros((6,4), dtype=np.float32),
            high=np.ones((6,4), dtype=np.float32)*np.array([1,4.0,6.0,1], dtype=np.float32),
            dtype=np.float32
        )
        
        # Observation space: 6 × SEQ_LEN × FEATURE_DIM
        self.observation_space = spaces.Box(
            low=-5, high=5, shape=(6,Config.SEQ_LEN,Config.FEATURE_DIM), dtype=np.float32
        )
        
        self.confidence_threshold = Config.CONFIDENCE_THRESHOLD_START
        self.ct_initial = Config.CONFIDENCE_THRESHOLD_START
        self.ct_final = Config.CONFIDENCE_THRESHOLD_END
        self.th_start = Config.THRESHOLD_TRANSITION_START
        self.th_end = Config.THRESHOLD_TRANSITION_END
        self.reset()
        self.last_summary_time = time.time()
        self.opened_trades_history = deque(maxlen=Config.TRADE_HISTORY_SIZE)
        self.closed_trades_history = deque(maxlen=Config.TRADE_HISTORY_SIZE)

    def reset(self):
        self.balance = Config.INITIAL_BALANCE
        self.equity = Config.INITIAL_BALANCE
        self.open_trades = {}
        self.pending_entries = []
        self.current_step = {s: Config.SEQ_LEN for s in self.symbols}
        self.trade_count = 0
        self.max_equity = self.balance
        self.total_trades = 0
        self.profitable_trades = 0
        self.total_pnl = 0
        self.peak_drawdown = 0
        return self._get_obs()

    def _get_obs(self):
        obs = []
        for s in self.symbols:
            bar = self.data_engine.get_sequence(s, self.current_step[s])
            if bar is None:
                feats = np.zeros((Config.SEQ_LEN, Config.FEATURE_DIM), dtype=np.float32)
            else:
                feats = bar['features']
            # Debug logging for feature dimensions
            if feats.shape[1] != Config.FEATURE_DIM:
                logger.warning(f"Unexpected feature dimension: {feats.shape} for {s}")
            obs.append(feats)
        return np.stack(obs, axis=0)

    def _calculate_stops(self, entry, dir, atr, slm, tpm, sym):
        pip = self.pip[sym]
        scaled = atr * 10
        ap = scaled / pip
        ap = np.clip(ap, Config.MIN_STOP_DISTANCE_PIPS, Config.MAX_ATR_PIPS)
        slp = ap * slm
        tpp = ap * tpm
        if dir == 1:
            return entry - slp * pip, entry + tpp * pip, slp
        return entry + slp * pip, entry - tpp * pip, slp

    def _position_size(self, sym, risk_amt, slp, entry):
        if sym == "XAUUSD":
            pv = Config.XAUUSD_PIP_VALUE
        elif "JPY" in sym:
            pv = 1000.0 / entry
        else:
            pv = 10.0
        lot = risk_amt / (slp * pv)
        if sym == "XAUUSD":
            lot *= Config.XAUUSD_LOT_MULTIPLIER
        return max(Config.MIN_LOT, min(lot, Config.MAX_LOT))
    
    # Increased news significance threshold
    def _is_significant_news(self, nc, sv):
        return nc > 5 and abs(sv) > 0.7

    def _calculate_risk_reward(self, entry, sl, tp, dir):
        r = abs(entry - sl)
        w = abs(tp - entry)
        return w / r if r > 0 else 0

    def _log_trade_summary(self):
        now = time.time()
        if now - self.last_summary_time < Config.PROGRESS_INTERVAL:
            return False
            
        wr = (self.profitable_trades / self.total_trades * 100) if self.total_trades else 0
        ap = (self.total_pnl / self.total_trades) if self.total_trades else 0
        dd = (self.max_equity - self.equity) / self.max_equity if self.max_equity else 0
        self.peak_drawdown = max(self.peak_drawdown, dd)
        
        s = f"\n╔═══════════════════════════╗"
        s += f"\n║      Trading Summary      ║"
        s += f"\n╠═══════════════════════════╣"
        s += f"\n║ Balance: ${self.balance:,.2f} | Equity: ${self.equity:,.2f} ║"
        s += f"\n║ Trades: {self.total_trades} | Win Rate: {wr:.1f}% ║"
        s += f"\n║ Avg PnL: ${ap:.2f} | Total PnL: ${self.total_pnl:.2f} ║"
        s += f"\n║ Peak DD: {self.peak_drawdown*100:.1f}% | Conf: {self.confidence_threshold:.2f} ║"
        s += f"\n╚═══════════════════════════╝"
        
        if self.opened_trades_history:
            s += "\n\n╔═══════════════════╗"
            s += "\n║    Open Trades    ║"
            s += "\n╠═══════════════════╣"
            for t in list(self.opened_trades_history)[-6:]:
                rr = self._calculate_risk_reward(t['entry'], t['sl'], t['tp'], t['direction'])
                s += f"\n║ {t['symbol']} {'LONG' if t['direction']==1 else 'SHORT'} @ {t['entry']:.5f}"
                s += f"\n║ Size: {t['size']:.2f} lots | R/R: {rr:.2f}:1"
                s += f"\n║ SL: {t['sl']:.5f} | TP: {t['tp']:.5f}"
                s += f"\n║ Risk: ${t['risk_amount']:,.2f} | Conf: {t['confidence']:.2f}"
                s += "\n╠═══════════════════╣"
        
        if self.closed_trades_history:
            s += "\n\n╔════════════════════╗"
            s += "\n║   Closed Trades    ║"
            s += "\n╠════════════════════╣"
            for t in list(self.closed_trades_history)[-6:]:
                rr = self._calculate_risk_reward(t['entry'], t['sl'], t['tp'], t['direction'])
                pnl_color = "+" if t['pnl'] >= 0 else ""
                s += f"\n║ {t['symbol']} {'LONG' if t['direction']==1 else 'SHORT'} | {t['reason']}"
                s += f"\n║ Entry: {t['entry']:.5f} | Exit: {t['exit']:.5f}"
                s += f"\n║ PnL: {pnl_color}${t['pnl']:.2f} | Bars: {t['duration']} | R/R: {rr:.2f}:1"
                s += f"\n║ Conf: {t['confidence']:.2f}"
                s += "\n╠════════════════════╣"
                
        logger.info(s)
        self.last_summary_time = now
        return True

    def step(self, actions):
        total_reward = 0
        done = False
        infos = {}
        obs = []
        
        # Process each symbol
        for i, s in enumerate(self.symbols):
            act = actions[i]
            bar = self.data_engine.get_sequence(s, self.current_step[s])
            if bar is None:
                continue
                
            curr, high, low, atr = bar['current_price'], bar['high'], bar['low'], bar['atr']
            nc, sv = bar['news_count'], bar['avg_sentiment']
            sig, slm, tpm, conf = act
            
            # Update equity for open trades
            self.equity = self.balance
            for tr in self.open_trades.values():
                if tr['symbol'] != s:
                    continue
                diff = curr - tr['entry']
                if s == "XAUUSD":
                    pnl = diff * tr['direction'] * tr['size'] * 100
                elif "JPY" in s:
                    pnl = diff * tr['direction'] * tr['size'] * 100000 / curr
                else:
                    pnl = diff * tr['direction'] * tr['size'] * 100000
                self.equity += pnl
            
            # Process pending entries
            next_open = bar['open']
            pend = [pe for pe in self.pending_entries if pe['symbol'] == s]
            for pe in pend:
                d = pe['direction']
                ep = next_open
                slp, tp, sp = self._calculate_stops(ep, d, pe['atr'], pe['sl_mult'], pe['tp_mult'], s)
                ra = pe['risk_frac'] * self.balance
                ls = self._position_size(s, ra, sp, ep)
                tr = {
                    'id': pe['id'], 'symbol': s, 'direction': d,
                    'entry': ep, 'sl': slp, 'tp': tp, 'size': ls,
                    'sl_pips': sp, 'confidence': pe['confidence'],
                    'risk_amount': ra, 'atr': pe['atr'],
                    'open_step': self.current_step[s]
                }
                self.open_trades[pe['id']] = tr
                self.opened_trades_history.append(tr.copy())
            self.pending_entries = [pe for pe in self.pending_entries if pe['symbol'] != s]
            
            # Queue new trades
            active_trades = len([t for t in self.open_trades.values() if t['symbol']==s])
            pending_trades = len([pe for pe in self.pending_entries if pe['symbol']==s])
            if (conf > self.confidence_threshold and
                active_trades + pending_trades < Config.MAX_OPEN_TRADES):
                rf = Config.MIN_RISK + conf * (Config.MAX_RISK - Config.MIN_RISK)
                pid = f"{s}-{self.trade_count}"
                self.pending_entries.append({
                    'id': pid, 'symbol': s, 'direction': 1 if sig > 0.5 else -1,
                    'atr': atr, 'sl_mult': slm, 'tp_mult': tpm,
                    'confidence': conf, 'risk_frac': rf
                })
                self.trade_count += 1
                total_reward += 0.01
                
            # Close trades
            for tid, tr in list(self.open_trades.items()):
                if tr['symbol'] != s or tr['open_step'] == self.current_step[s]:
                    continue
                    
                ep = None
                rsn = ""
                d = tr['direction']
                
                # Skip SL/TP check in the same bar the trade was opened
                if tr['open_step'] < self.current_step[s]:
                    if (d == 1 and low <= tr['sl']) or (d == -1 and high >= tr['sl']):
                        ep, rsn = tr['sl'], "SL"
                    elif (d == 1 and high >= tr['tp']) or (d == -1 and low <= tr['tp']):
                        ep, rsn = tr['tp'], "TP"
                
                # News-based closing (with higher threshold)
                if ep is None and self._is_significant_news(nc, sv):
                    ep, rsn = curr, "NEWS"
                    
                if ep is not None:
                    diff = ep - tr['entry']
                    if s == "XAUUSD":
                        pnl = diff * d * tr['size'] * 100
                    elif "JPY" in s:
                        pnl = diff * d * tr['size'] * 100000 / ep
                    else:
                        pnl = diff * d * tr['size'] * 100000
                    pnl -= Config.COMMISSION * tr['size']
                    self.balance += pnl
                    self.total_pnl += pnl
                    self.total_trades += 1
                    if pnl > 0:
                        self.profitable_trades += 1
                    
                    # Reward shaping
                    reward_multiplier = 1.5 if rsn == "TP" else 0.8 if rsn == "SL" else 1.0
                    total_reward += pnl * reward_multiplier / 1000
                    rr = self._calculate_risk_reward(tr['entry'], tr['sl'], tr['tp'], d)
                    total_reward += min(rr, 5.0) * 0.01
                    
                    cl = tr.copy()
                    cl.update({
                        'exit': ep, 'reason': rsn, 'pnl': pnl,
                        'duration': self.current_step[s] - tr['open_step']
                    })
                    self.closed_trades_history.append(cl)
                    del self.open_trades[tid]
            
            # Record observation
            obs.append(bar['features'])
            infos[s] = {
                'balance': self.balance,
                'equity': self.equity,
                'symbol': s,
                'drawdown': (self.max_equity - self.equity) / self.max_equity if self.max_equity else 0
            }
            
        # Advance to next time step
        for s in self.symbols:
            self.current_step[s] += 1
            if self.current_step[s] >= len(self.data_engine.data[s]) - 1:
                self.current_step[s] = Config.SEQ_LEN
                
        self.max_equity = max(self.max_equity, self.equity)
        total_reward -= 0.001  # Small time penalty
        
        # Check termination conditions
        balance_condition = self.balance < Config.INITIAL_BALANCE * 0.7
        dd_condition = (self.max_equity - self.equity) / self.max_equity > Config.DAILY_DD_LIMIT
        done = balance_condition or dd_condition
        
        # Log trade summary every 30 seconds
        self._log_trade_summary()
        
        return np.stack(obs, 0), total_reward, done, infos

    def update_confidence_threshold(self, progress):
        if progress < self.th_start:
            self.confidence_threshold = self.ct_initial
        elif progress > self.th_end:
            self.confidence_threshold = self.ct_final
        else:
            t = (progress - self.th_start) / (self.th_end - self.th_start)
            self.confidence_threshold = self.ct_initial + (self.ct_final - self.ct_initial) * t

# ================ NEURAL NETWORK ================
class ForexPolicy(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # Reduced hidden size from 256 to 128
        self.lstm = nn.LSTM(input_dim, 128, 2, batch_first=True, bidirectional=True)
        self.feature_net = nn.Sequential(
            nn.Linear(256, 128),  # Adjusted for bidirectional output (128*2=256)
            nn.LeakyReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, Config.POLICY_DIM),  # POLICY_DIM reduced to 128
            nn.Tanh()
        )
        self.actor_signal = nn.Linear(Config.POLICY_DIM, 1)
        self.actor_sl = nn.Linear(Config.POLICY_DIM, 1)
        self.actor_tp = nn.Linear(Config.POLICY_DIM, 1)
        self.actor_conf = nn.Linear(Config.POLICY_DIM, 1)
        self.critic = nn.Sequential(
            nn.Linear(Config.POLICY_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LSTM):
            for n, p in m.named_parameters():
                if 'weight_ih' in n:
                    nn.init.xavier_uniform_(p.data)
                elif 'weight_hh' in n:
                    nn.init.orthogonal_(p.data)
                elif 'bias' in n:
                    nn.init.constant_(p.data, 0)

    def forward(self, x):
        # Handle both single and batched inputs
        if x.dim() == 3:
            x = x.unsqueeze(0)
        batch_size, num_symbols, seq_len, feat_dim = x.size()
        x = x.view(batch_size * num_symbols, seq_len, feat_dim)  # FIXED: Proper reshaping
        out, _ = self.lstm(x)
        f = self.feature_net(out[:, -1, :])
        sig = torch.sigmoid(self.actor_signal(f))
        sl = 1.0 + 3.0 * torch.sigmoid(self.actor_sl(f))
        tp = 1.5 + 4.5 * torch.sigmoid(self.actor_tp(f))
        conf = torch.sigmoid(self.actor_conf(f))
        val = self.critic(f)
        actions = torch.cat([sig, sl, tp, conf], dim=-1)
        actions = actions.view(batch_size, num_symbols, 4)
        val = val.view(batch_size, num_symbols, 1).mean(dim=1)
        return actions, val

    def act(self, obs):
        with torch.no_grad():
            t = torch.FloatTensor(obs).to(Config.DEVICE)
            acts, val = self.forward(t)
            acts = acts.squeeze(0)
            # Exploration noise
            noise = torch.randn_like(acts) * 0.1
            na = acts + noise
            na[:, 0].clamp_(0, 1)
            na[:, 1].clamp_(1.0, 4.0)
            na[:, 2].clamp_(1.5, 6.0)
            na[:, 3].clamp_(0, 1)
            return na.cpu().numpy(), val.item()

# ================ PPO TRAINER ================
class PPOTrainer:
    def __init__(self, env):
        self.env = env
        self.policy = ForexPolicy(Config.FEATURE_DIM).to(Config.DEVICE)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=Config.LR)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.TIMESTEPS, eta_min=1e-6)
        self.memory = deque(maxlen=Config.BATCH_SIZE)
        self.writer = SummaryWriter(Config.LOG_DIR)
        self.step_count = 0
        self.episode_count = 0
        self.best_balance = Config.INITIAL_BALANCE
        self.Transition = namedtuple('Transition', ['obs', 'action', 'value', 'reward', 'done'])

    def train(self):
        obs = self.env.reset()
        ep_reward = 0
        start = time.time()
        while self.step_count < Config.TIMESTEPS:
            acts, val = self.policy.act(obs)
            next_obs, rew, done, info = self.env.step(acts)
            prog = self.step_count / Config.TIMESTEPS
            self.env.update_confidence_threshold(prog)
            self.memory.append(self.Transition(obs, acts, val, rew, done))
            obs = next_obs if not done else self.env.reset()
            self.step_count += 1
            ep_reward += rew
            
            # Logging
            if self.step_count % 100 == 0:
                self.writer.add_scalar("Balance", info[Config.SYMBOLS[0]]['balance'], self.step_count)
                self.writer.add_scalar("Equity", info[Config.SYMBOLS[0]]['equity'], self.step_count)
                self.writer.add_scalar("Reward", rew, self.step_count)
                self.writer.add_scalar("Drawdown", info[Config.SYMBOLS[0]]['drawdown'], self.step_count)
                self.writer.add_scalar("Conf_Th", self.env.confidence_threshold, self.step_count)
                
            if done:
                self.episode_count += 1
                self.writer.add_scalar("Episode/Reward", ep_reward, self.episode_count)
                self.writer.add_scalar("Episode/Balance", info[Config.SYMBOLS[0]]['balance'], self.episode_count)
                self.writer.add_scalar("Episode/Drawdown", info[Config.SYMBOLS[0]]['drawdown'], self.episode_count)
                ep_reward = 0
                
            # Policy update
            if len(self.memory) >= Config.BATCH_SIZE:
                self.update_policy()
                
            # Model saving
            if self.step_count % Config.SAVE_INTERVAL == 0:
                bal = info[Config.SYMBOLS[0]]['balance']
                if bal > self.best_balance:
                    self.best_balance = bal
                    self.save_model(best=True)
                else:
                    self.save_model()
                    
            # Progress reporting
            if self.step_count % 1000 == 0:
                elapsed = time.time() - start
                spd = self.step_count / elapsed
                logger.info(f"Step:{self.step_count}/{Config.TIMESTEPS}|Bal${info[Config.SYMBOLS[0]]['balance']:,.2f}|Speed{spd:.1f}sps|DD{info[Config.SYMBOLS[0]]['drawdown']*100:.1f}%")
                
        self.save_model(final=True)

    def update_policy(self):
        # Memory cleanup before update
        gc.collect()
        if Config.DEVICE == "cuda":
            torch.cuda.empty_cache()
            
        # Unpack memory
        obs_b = torch.tensor([t.obs for t in self.memory], dtype=torch.float32).to(Config.DEVICE)
        acts_b = torch.tensor([t.action for t in self.memory], dtype=torch.float32).to(Config.DEVICE)
        vals_b = torch.tensor([t.value for t in self.memory], dtype=torch.float32).to(Config.DEVICE)
        rews = torch.tensor([t.reward for t in self.memory], dtype=torch.float32).to(Config.DEVICE)
        dns = torch.tensor([t.done for t in self.memory], dtype=torch.float32).to(Config.DEVICE)
        
        # Compute returns
        returns = torch.zeros_like(rews)
        R = 0
        for i in reversed(range(len(rews))):
            R = rews[i] + Config.GAMMA * R * (1 - dns[i])
            returns[i] = R
            
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Policy updates
        self.optimizer.zero_grad()
        
        # Gradient accumulation
        mini_batch_size = Config.BATCH_SIZE // Config.GRAD_ACCUM_STEPS
        total_loss = 0
        
        for epoch in range(Config.N_EPOCHS):
            indices = torch.randperm(Config.BATCH_SIZE)
            for i in range(Config.GRAD_ACCUM_STEPS):
                start = i * mini_batch_size
                end = start + mini_batch_size
                idx = indices[start:end]
                
                obs_mini = obs_b[idx]
                acts_mini = acts_b[idx]
                vals_mini = vals_b[idx]
                returns_mini = returns[idx]
                
                new_acts, new_vals = self.policy(obs_mini)
                new_vals = new_vals.squeeze()
                
                # Compute advantages
                adv = returns_mini - vals_mini
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                
                # Policy loss
                ratio = torch.exp(-0.5 * ((acts_mini - new_acts) ** 2).sum(-1))
                # FIX: Reshape adv for broadcasting
                adv_expanded = adv.unsqueeze(1)  # Shape [batch, 1]
                s1 = ratio * adv_expanded
                s2 = torch.clamp(ratio, 1-Config.CLIP_PARAM, 1+Config.CLIP_PARAM) * adv_expanded
                policy_loss = -torch.min(s1, s2).mean()
                
                # Value loss
                value_loss_unclipped = (new_vals - returns_mini) ** 2
                value_clipped = vals_mini + torch.clamp(
                    new_vals - vals_mini, -Config.CLIP_PARAM, Config.CLIP_PARAM)
                value_loss_clipped = (value_clipped - returns_mini) ** 2
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
                
                # Entropy bonus
                entropy = -0.5 * torch.log(2 * torch.tensor(math.pi)) - 0.5 * (acts_mini - new_acts).pow(2).mean()
                entropy_bonus = Config.ENTROPY_COEF * entropy
                
                # Total loss
                loss = policy_loss + Config.VALUE_COEF * value_loss - entropy_bonus
                loss = loss / Config.GRAD_ACCUM_STEPS
                loss.backward()
                total_loss += loss.item()
                
        # Optimization step
        nn.utils.clip_grad_norm_(self.policy.parameters(), Config.GRAD_CLIP)
        self.optimizer.step()
        self.scheduler.step()
        self.memory.clear()
        
        # Log loss
        self.writer.add_scalar("Loss/Total", total_loss, self.step_count)
        
    def save_model(self, best=False, final=False):
        os.makedirs(Config.MODEL_DIR, exist_ok=True)
        if best:
            path = os.path.join(Config.MODEL_DIR, "forex_ai_best.pth")
        elif final:
            path = os.path.join(Config.MODEL_DIR, "forex_ai_final.pth")
        else:
            path = os.path.join(Config.MODEL_DIR, f"forex_ai_step_{self.step_count}.pth")
        torch.save({
            'step': self.step_count,
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }, path)
        logger.info(f"Saved model checkpoint: {path}")

if __name__ == "__main__":
    logger.info("Starting Forex AI Training")
    try:
        data_engine = ForexDataEngine()
        env = ForexTradingEnv(data_engine)
        trainer = PPOTrainer(env)
        trainer.train()
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise
    logger.info("Training completed successfully")
