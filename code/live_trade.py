#!/usr/bin/env python3

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import requests
import pandas_ta as ta
import random
import logging
import os
import json
from datetime import datetime, timedelta
from collections import deque
from dateutil.parser import parse
import gc
import psutil

# ================ CONFIGURATION ================
class Config:
    # File Paths
    BASE_DIR = "C:/FPFX"
    MODEL_DIR = f"{BASE_DIR}/model"
    LOG_DIR = f"{BASE_DIR}/logs"
    TRADE_LOG_DIR = f"{BASE_DIR}/trade_logs"
    TRADE_MEMORY_DIR = f"{BASE_DIR}/trade_memory"
    MODEL_FILE = f"{MODEL_DIR}/forex_ai_best.pth"
    MODEL_BACKUP = f"{MODEL_DIR}/forex_ai_backup.pth"

    # MT5 Login Details
    MT5_ACCOUNT = 52413066
    MT5_PASSWORD = "@7XenGbV5vs5$a"
    MT5_SERVER = "ICMarketsSC-Demo"

    # Trading Parameters
    SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "XAUUSD"]
    SYMBOL_NEWS_KEYWORDS = {
        "EURUSD": ["EURUSD", "Euro Dollar", "Euro USD", "EUR USD"],
        "GBPUSD": ["GBPUSD", "Cable", "Pound Dollar", "GBP USD"],
        "USDJPY": ["USDJPY", "Yen", "Dollar Yen", "USD JPY"],
        "AUDUSD": ["AUDUSD", "Aussie", "Australian Dollar", "AUD USD"],
        "USDCAD": ["USDCAD", "Loonie", "Canadian Dollar", "USD CAD", "Oil", "Crude"],
        "XAUUSD": ["XAUUSD", "Gold", "Gold USD", "XAU USD"]
    }
    PIP_SIZES = {
        "EURUSD": 0.0001,
        "GBPUSD": 0.0001,
        "USDJPY": 0.01,
        "AUDUSD": 0.0001,
        "USDCAD": 0.0001,
        "XAUUSD": 0.1
    }
    MIN_STOP_PIPS = {
        "EURUSD": 12,
        "GBPUSD": 15,
        "USDJPY": 10,
        "AUDUSD": 10,
        "USDCAD": 12,
        "XAUUSD": 25
    }
    MIN_RR_RATIO = 1.3  # Minimum risk/reward ratio
    INITIAL_BALANCE = None
    BALANCE_LIMIT = None
    CONFIDENCE_THRESHOLD = 0.6
    HIGH_CONFIDENCE_THRESHOLD = 0.8
    MEDIUM_CONFIDENCE_INTERVAL = 12
    HIGH_CONFIDENCE_INTERVAL = 3
    MIN_RISK = 0.005
    MAX_RISK = 0.015
    COMMISSION = 0.0002
    NEWS_SENTIMENT_THRESHOLD = 0.7
    NEWS_CHECK_INTERVAL = 120
    NEWS_LOOKBACK_HOURS = 6
    BAR_TIMEFRAME = mt5.TIMEFRAME_M5
    MAX_OPEN_TRADES = 3
    SL_MULTIPLIER_RANGE = (1.5, 2.5)  # Updated range
    TP_MULTIPLIER_RANGE = (2.0, 3.5)  # Updated range
    SEQ_LEN = 24
    INITIAL_DATA_WINDOW = 300
    TRAILING_ATR_MULTIPLIER = 1.5

    # Market Hours (UTC)
    MARKET_OPEN_HOUR = 0
    MARKET_CLOSE_HOUR = 21
    MARKET_DAYS = [0, 1, 2, 3, 4]

    # EODHD API
    EODHD_API_KEY = "684d76d81e83b0.88582643"

    # Technical Indicators
    FEATURE_COLS = [
        'open', 'high', 'low', 'close', 'RSI_14', 'BB_%B', 'BB_width', 'ATR_14',
        'STOCH_%K', 'STOCH_%D', 'MACD_line', 'MACD_signal', 'MACD_hist',
        'volatility', 'momentum', 'ADX_14', 'DMP_14', 'DMN_14', 'SMA_50', 'SMA_200',
        'TRIX_15', 'volume_ma', 'volume_pct', 'Regime0', 'Regime1', 'Regime2', 'Regime3',
        'hour', 'day_of_week', 'news_count', 'avg_sentiment'
    ]

    # Continuous Learning Parameters
    LEARNING_RATE = 3e-6
    LOSS_PENALTY = 1.5
    WIN_REWARD = 0.8
    MAX_GRAD_NORM = 1.0
    PASSIVE_LEARN_INTERVAL = 3600
    MEMORY_CAPACITY = 2000
    PRIORITY_ALPHA = 0.6
    ADAPTIVE_LR = True
    EWC_LAMBDA = 0.4
    ARCHITECTURE_SEARCH_INTERVAL = 86400

    # Correlation Matrix
    CORRELATION_MATRIX = {
        'EURUSD': {'EURUSD': 1.0, 'GBPUSD': 0.89, 'USDJPY': -0.78, 'AUDUSD': 0.92, 'USDCAD': -0.85, 'XAUUSD': -0.45},
        'GBPUSD': {'EURUSD': 0.89, 'GBPUSD': 1.0, 'USDJPY': -0.82, 'AUDUSD': 0.85, 'USDCAD': -0.79, 'XAUUSD': -0.52},
        'USDJPY': {'EURUSD': -0.78, 'GBPUSD': -0.82, 'USDJPY': 1.0, 'AUDUSD': -0.76, 'USDCAD': 0.88, 'XAUUSD': 0.38},
        'AUDUSD': {'EURUSD': 0.92, 'GBPUSD': 0.85, 'USDJPY': -0.76, 'AUDUSD': 1.0, 'USDCAD': -0.81, 'XAUUSD': -0.41},
        'USDCAD': {'EURUSD': -0.85, 'GBPUSD': -0.79, 'USDJPY': 0.88, 'AUDUSD': -0.81, 'USDCAD': 1.0, 'XAUUSD': 0.35},
        'XAUUSD': {'EURUSD': -0.45, 'GBPUSD': -0.52, 'USDJPY': 0.38, 'AUDUSD': -0.41, 'USDCAD': 0.35, 'XAUUSD': 1.0}
    }

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================ LOGGING SETUP ================
# Main logger for system messages
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"{Config.LOG_DIR}/forex_ai_system.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ForexAI_System")

# Trade-specific logger
trade_logger = logging.getLogger("ForexAI_Trades")
trade_logger.setLevel(logging.INFO)
os.makedirs(Config.TRADE_LOG_DIR, exist_ok=True)
trade_handler = logging.FileHandler(f"{Config.TRADE_LOG_DIR}/trade_execution.log", encoding='utf-8')
trade_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
trade_logger.addHandler(trade_handler)
trade_logger.propagate = False

# ================ GLOBAL VARIABLES ================
trade_outcomes = []
trade_memory = deque(maxlen=Config.MEMORY_CAPACITY)
market_regime = "neutral"
last_regime_update = 0
fisher_matrix = None
previous_params = None
model = None
learner = None

# ================ CONTINUOUS LEARNING IMPROVEMENTS ================
class ContinuousLearner:
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', patience=100)
        self.importance_weights = {}
        self.last_arch_search = time.time()
        self.fisher_matrix = None
        self.previous_params = None

    def compute_fisher_matrix(self):
        fisher = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                fisher[name] = param.grad.data.clone().pow(2)
        return fisher

    def elastic_weight_consolidation(self):
        if not hasattr(self, 'fisher_matrix') or not self.fisher_matrix:
            return 0
        ewc_loss = 0
        for name, param in self.model.named_parameters():
            if name in self.fisher_matrix:
                ewc_loss += (self.fisher_matrix[name] * 
                            (param - self.previous_params[name]).pow(2)).sum()
        return Config.EWC_LAMBDA * ewc_loss

    def prioritized_experience_replay(self, batch_size=32):
        priorities = np.array([abs(x['outcome'] - x['confidence']) for x in trade_memory])
        probs = priorities ** Config.PRIORITY_ALPHA
        probs /= probs.sum()
        indices = np.random.choice(len(trade_memory), batch_size, p=probs)
        batch = [trade_memory[i] for i in indices]
        weights = (len(trade_memory) * probs[indices]) ** (-Config.PRIORITY_ALPHA)
        weights /= weights.max()
        return batch, torch.FloatTensor(weights).to(Config.DEVICE)

    def adapt_architecture(self):
        if time.time() - self.last_arch_search < Config.ARCHITECTURE_SEARCH_INTERVAL:
            return
        self.last_arch_search = time.time()
        win_rate = sum(trade_outcomes[-100:])/min(100, len(trade_outcomes))
        if win_rate < 0.52 and not hasattr(self.model, 'adaptive_layer1'):
            self.model.adaptive_layer1 = nn.Linear(128, 128).to(Config.DEVICE)
            logger.info("Added adaptive layer 1 to model")
        elif win_rate > 0.58 and hasattr(self.model, 'adaptive_layer1'):
            del self.model.adaptive_layer1
            logger.info("Removed adaptive layer from model")

# ================ ENHANCED MODEL ================
class EnhancedForexPolicy(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, 128, 2, batch_first=True, bidirectional=True)
        self.feature_net = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 128),
            nn.Tanh()
        )
        self.actor_signal = nn.Linear(128, 1)
        self.actor_sl = nn.Linear(128, 1)
        self.actor_tp = nn.Linear(128, 1)
        self.actor_conf = nn.Linear(128, 1)
        self.critic = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        batch_size, num_symbols, seq_len, feat_dim = x.size()
        x = x.view(batch_size * num_symbols, seq_len, feat_dim)
        out, _ = self.lstm(x)
        f = self.feature_net(out[:, -1, :])
        if hasattr(self, 'adaptive_layer1'):
            f = self.adaptive_layer1(f)
        sig = torch.sigmoid(self.actor_signal(f))
        sl = 1.0 + 3.0 * torch.sigmoid(self.actor_sl(f))  # Scales to 1.0-4.0
        tp = 1.5 + 4.5 * torch.sigmoid(self.actor_tp(f))  # Scales to 1.5-6.0
        conf = torch.sigmoid(self.actor_conf(f))
        val = self.critic(f)
        actions = torch.cat([sig, sl, tp, conf], dim=-1)
        actions = actions.view(batch_size, num_symbols, 4)
        val = val.view(batch_size, num_symbols, 1).mean(dim=1)
        return actions, val

# ================ MT5 INITIALIZATION ================
def initialize_mt5():
    if not mt5.initialize():
        raise Exception("MT5 initialization failed")
    authorized = mt5.login(
        login=Config.MT5_ACCOUNT,
        password=Config.MT5_PASSWORD,
        server=Config.MT5_SERVER
    )
    if not authorized:
        raise Exception(f"Failed to connect to account #{Config.MT5_ACCOUNT}, error: {mt5.last_error()}")
    account_info = mt5.account_info()
    if account_info is None:
        raise Exception("Failed to get account info")
    Config.INITIAL_BALANCE = account_info.balance
    Config.BALANCE_LIMIT = Config.INITIAL_BALANCE * 0.8
    logger.info(f"MT5 Connected. Balance: ${Config.INITIAL_BALANCE:.2f}, Stop Limit: ${Config.BALANCE_LIMIT:.2f}")

# ================ MARKET HOURS CHECK ================
def is_market_open():
    now = datetime.utcnow()
    current_hour = now.hour
    current_day = now.weekday()
    if current_day not in Config.MARKET_DAYS:
        return False
    if Config.MARKET_OPEN_HOUR <= current_hour < Config.MARKET_CLOSE_HOUR:
        return True
    return False

# ================ DATA PROCESSING ================
def fetch_latest_data(symbol, timeframe=Config.BAR_TIMEFRAME, window=Config.INITIAL_DATA_WINDOW):
    if not mt5.symbol_select(symbol, True):
        logger.error(f"Failed to select {symbol}, error: {mt5.last_error()}")
        return None
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, window)
    if rates is None or len(rates) == 0:
        logger.error(f"Failed to fetch rates for {symbol}, error: {mt5.last_error()}")
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

def get_h1_atr(symbol):
    """Get true ATR in pips for H1 timeframe"""
    df = fetch_latest_data(symbol, mt5.TIMEFRAME_H1, 50)
    if df is None or len(df) < 15:
        logger.error(f"Failed to get H1 data for {symbol}")
        return None
    
    atr = ta.atr(df['high'], df['low'], df['close'], length=14)
    if atr is None or atr.iloc[-1] is None:
        logger.error(f"ATR calculation failed for {symbol} H1")
        return None
        
    pip_size = Config.PIP_SIZES.get(symbol, 0.0001)
    atr_pips = atr.iloc[-1] / pip_size
    logger.info(f"H1 ATR for {symbol}: {atr.iloc[-1]:.6f} = {atr_pips:.1f} pips")
    return atr_pips

def calculate_indicators(df, symbol, include_news=True):
    if df is None or len(df) == 0:
        logger.error(f"Empty dataframe for {symbol}")
        return None
    try:
        df['RSI_14'] = ta.rsi(df['close'], length=14)
        if df['RSI_14'] is None:
            df['RSI_14'] = 50.0
        bb = ta.bbands(df['close'], length=20, std=2)
        if bb is not None:
            df['BB_upper'] = bb['BBU_20_2.0'].fillna(df['close'])
            df['BB_middle'] = bb['BBM_20_2.0'].fillna(df['close'])
            df['BB_lower'] = bb['BBL_20_2.0'].fillna(df['close'])
            df['BB_%B'] = (df['close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
            df['BB_width'] = df['BB_upper'] - df['BB_lower']
        else:
            df['BB_upper'] = df['close']
            df['BB_middle'] = df['close']
            df['BB_lower'] = df['close']
            df['BB_%B'] = 0.5
            df['BB_width'] = 0
        atr = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['ATR_14'] = atr.fillna(0) if atr is not None else 0
        stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3)
        if stoch is not None:
            df['STOCH_%K'] = stoch['STOCHk_14_3_3'].fillna(50)
            df['STOCH_%D'] = stoch['STOCHd_14_3_3'].fillna(50)
        else:
            df['STOCH_%K'] = 50
            df['STOCH_%D'] = 50
        macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
        if macd is not None:
            df['MACD_line'] = macd['MACD_12_26_9'].fillna(0)
            df['MACD_signal'] = macd['MACDs_12_26_9'].fillna(0)
            df['MACD_hist'] = macd['MACDh_12_26_9'].fillna(0)
        else:
            df['MACD_line'] = 0
            df['MACD_signal'] = 0
            df['MACD_hist'] = 0
        df['volatility'] = df['close'].rolling(14).std().fillna(0)
        df['momentum'] = df['close'].pct_change(14).fillna(0)
        adx = ta.adx(df['high'], df['low'], df['close'], length=14)
        if adx is not None:
            df['ADX_14'] = adx['ADX_14'].fillna(25)
            df['DMP_14'] = adx['DMP_14'].fillna(0)
            df['DMN_14'] = adx['DMN_14'].fillna(0)
        else:
            df['ADX_14'] = 25
            df['DMP_14'] = 0
            df['DMN_14'] = 0
        sma_50 = ta.sma(df['close'], length=50)
        df['SMA_50'] = sma_50.fillna(df['close']) if sma_50 is not None else df['close']
        sma_200 = ta.sma(df['close'], length=200)
        df['SMA_200'] = sma_200.fillna(df['close']) if sma_200 is not None else df['close']
        trix = ta.trix(df['close'], length=15)
        if trix is not None:
            if isinstance(trix, pd.DataFrame):
                df['TRIX_15'] = trix.iloc[:, 0].fillna(0)
            else:
                df['TRIX_15'] = trix.fillna(0)
        else:
            df['TRIX_15'] = 0
        if 'real_volume' in df.columns:
            df['volume_ma'] = df['real_volume'].rolling(14).mean().fillna(0)
            df['volume_pct'] = df['real_volume'].pct_change().fillna(0)
        else:
            df['volume_ma'] = 0
            df['volume_pct'] = 0
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df[["Regime0", "Regime1", "Regime2", "Regime3"]] = 0
        if include_news:
            sentiment, count = get_eodhd_sentiment(symbol)
            df["news_count"] = count
            df["avg_sentiment"] = sentiment
        else:
            df["news_count"] = 0
            df["avg_sentiment"] = 0
        df.fillna(0, inplace=True)
        return df
    except Exception as e:
        logger.error(f"Indicator calculation failed for {symbol}: {str(e)}")
        return None

# ================ NEWS MONITORING ================
def get_eodhd_sentiment(symbol, lookback_hours=6):
    now = datetime.utcnow()
    time_threshold = now - timedelta(hours=lookback_hours)
    symbol_map = {
        "EURUSD": "EURUSD.FOREX",
        "GBPUSD": "GBPUSD.FOREX",
        "USDJPY": "USDJPY.FOREX",
        "AUDUSD": "AUDUSD.FOREX",
        "USDCAD": "USDCAD.FOREX",
        "XAUUSD": "XAUUSD.FOREX"
    }
    api_symbol = symbol_map.get(symbol)
    if api_symbol is None:
        logger.error(f"No API symbol mapping found for {symbol}")
        return 0, 0
    url = f"https://eodhd.com/api/news?s={api_symbol}&offset=0&limit=100&api_token={Config.EODHD_API_KEY}&fmt=json"
    all_articles = []
    try:
        logger.info(f"Fetching news for {symbol}")
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            logger.error(f"News API Error: Status {response.status_code} for {api_symbol}")
            return 0, 0
        data = response.json()
        if isinstance(data, list):
            articles = data
            logger.info(f"Received direct articles list ({len(articles)} items)")
        elif isinstance(data, dict) and 'data' in data and isinstance(data['data'], list):
            articles = data['data']
            logger.info(f"Received articles in 'data' field ({len(articles)} items)")
        elif isinstance(data, dict) and 'error' in data:
            logger.error(f"News API Error: {data.get('error', 'Unknown error')}")
            return 0, 0
        else:
            logger.error(f"Unexpected API response format: {type(data)}")
            return 0, 0
        if not articles:
            logger.info(f"No articles returned for {api_symbol}")
            return 0, 0
        articles_within_threshold = 0
        for article in articles:
            try:
                if 'date' not in article:
                    logger.debug(f"Article missing date field: {article.get('title', 'Untitled')}")
                    continue
                date_str = article['date']
                try:
                    article_time = parse(date_str).replace(tzinfo=None)
                except Exception as e:
                    logger.debug(f"Error parsing article time: {str(e)}")
                    continue
                if article_time >= time_threshold:
                    all_articles.append(article)
                    articles_within_threshold += 1
            except Exception as e:
                logger.debug(f"Error processing article: {str(e)}")
                continue
        logger.info(f"Found {articles_within_threshold} articles within last {lookback_hours} hours for {symbol}")
        if not all_articles:
            logger.info(f"No news articles found for {symbol} in last {lookback_hours} hours")
            return 0, 0
        sentiments = []
        valid_articles = 0
        for article in all_articles:
            try:
                sentiment_score = None
                if 'sentiment' in article and isinstance(article['sentiment'], (int, float)):
                    sentiment_score = float(article['sentiment'])
                elif 'sentiment' in article and isinstance(article['sentiment'], str):
                    try:
                        sentiment_score = float(article['sentiment'])
                    except ValueError:
                        pass
                if sentiment_score is None and 'sentiment' in article and isinstance(article['sentiment'], dict):
                    sentiment_data = article['sentiment']
                    if 'polarity' in sentiment_data:
                        try:
                            sentiment_score = float(sentiment_data['polarity'])
                        except (ValueError, TypeError):
                            pass
                if sentiment_score is None and all(key in sentiment_data for key in ['pos', 'neg', 'neu']):
                    try:
                        pos = float(sentiment_data.get('pos', 0))
                        neg = float(sentiment_data.get('neg', 0))
                        neu = float(sentiment_data.get('neu', 0))
                        if pos + neg + neu > 0:
                            sentiment_score = (pos - neg) / (pos + neg + neu)
                    except (ValueError, TypeError):
                        pass
                if sentiment_score is not None and -1 <= sentiment_score <= 1:
                    sentiments.append(sentiment_score)
                    valid_articles += 1
            except Exception as e:
                logger.debug(f"Error processing article sentiment: {str(e)}")
                continue
        if not sentiments:
            logger.info(f"No usable sentiment data for {symbol} (found {len(all_articles)} articles)")
            return 0, len(all_articles)
        avg_sentiment = np.mean(sentiments) if sentiments else 0
        logger.info(f"News Check: {symbol} | Sentiment: {avg_sentiment:.2f} | Articles: {valid_articles}/{len(all_articles)} | Lookback: {lookback_hours}h")
        return avg_sentiment, len(all_articles)
    except Exception as e:
        logger.error(f"News API Failed for {api_symbol}: {str(e)}")
        return 0, 0

def check_recent_news():
    if not is_market_open():
        return
    logger.info("---------------------------------- Performing scheduled news check ----------------------------------")
    logger.info(f"Checking for news with sentiment threshold: {Config.NEWS_SENTIMENT_THRESHOLD}")
    for symbol in Config.SYMBOLS:
        sentiment, count = get_eodhd_sentiment(symbol, lookback_hours=5/60)
        logger.info(f"News sentiment for {symbol}: {sentiment:.2f} (count: {count})")
        if abs(sentiment) > Config.NEWS_SENTIMENT_THRESHOLD:
            logger.warning(f"Strong news sentiment detected for {symbol}: {sentiment:.2f}")
            positions = mt5.positions_get(symbol=symbol)
            if positions is None:
                logger.error(f"Failed to check positions for {symbol}, error: {mt5.last_error()}")
                continue
            if positions:
                for position in positions:
                    if (position.type == mt5.ORDER_TYPE_BUY and sentiment < -Config.NEWS_SENTIMENT_THRESHOLD) or \
                       (position.type == mt5.ORDER_TYPE_SELL and sentiment > Config.NEWS_SENTIMENT_THRESHOLD):
                        logger.info(f"Closing position {position.ticket} due to adverse news sentiment")
                        close_trade(position.ticket, "BREAKING_NEWS")
    logger.info("---------------------------------- Completed news check ----------------------------------")

# ================ DYNAMIC THRESHOLD FUNCTIONS ================
def auto_adjust_thresholds():
    if len(trade_outcomes) < 20:
        return
    win_rate = sum(trade_outcomes)/len(trade_outcomes)
    logger.info(f"Current win rate: {win_rate:.2%} | Targets: High={Config.HIGH_CONFIDENCE_THRESHOLD:.2f} Med={Config.CONFIDENCE_THRESHOLD:.2f}")
    if win_rate < Config.WIN_RATE_TARGET:
        Config.HIGH_CONFIDENCE_THRESHOLD = min(
            Config.MAX_HIGH_THRESH,
            Config.HIGH_CONFIDENCE_THRESHOLD + Config.THRESHOLD_SENSITIVITY
        )
        Config.CONFIDENCE_THRESHOLD = min(
            Config.MAX_MED_THRESH,
            Config.CONFIDENCE_THRESHOLD + Config.THRESHOLD_SENSITIVITY
        )
        logger.info(f"Increasing thresholds to High={Config.HIGH_CONFIDENCE_THRESHOLD:.2f} Med={Config.CONFIDENCE_THRESHOLD:.2f}")
    else:
        Config.HIGH_CONFIDENCE_THRESHOLD = max(
            Config.MIN_HIGH_THRESH,
            Config.HIGH_CONFIDENCE_THRESHOLD - Config.THRESHOLD_SENSITIVITY
        )
        Config.CONFIDENCE_THRESHOLD = max(
            Config.MIN_MED_THRESH,
            Config.CONFIDENCE_THRESHOLD - Config.THRESHOLD_SENSITIVITY
        )
        logger.info(f"Decreasing thresholds to High={Config.HIGH_CONFIDENCE_THRESHOLD:.2f} Med={Config.CONFIDENCE_THRESHOLD:.2f}")

# ================ MODEL INITIALIZATION ================
def initialize_model():
    global model, learner
    map_location = None if torch.cuda.is_available() else torch.device('cpu')
    try:
        checkpoint = torch.load(Config.MODEL_FILE, map_location=map_location)
        model = EnhancedForexPolicy(len(Config.FEATURE_COLS)).to(Config.DEVICE)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Loaded model from checkpoint")
        else:
            model.load_state_dict(checkpoint)
            logger.info("Loaded standalone model weights")
        model.float()
        model.train()
        learner = ContinuousLearner(model)
        if 'fisher_matrix' in checkpoint:
            learner.fisher_matrix = checkpoint['fisher_matrix']
            learner.previous_params = checkpoint['previous_params']
            logger.info("Loaded EWC parameters")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

# ================ LIVE LEARNING ================
def passive_learn(features_tensor, symbol=None):
    try:
        learner.optimizer.zero_grad()
        actions, _ = model(features_tensor)
        signal, sl_mult, tp_mult, confidence = actions[0, 0]
        target_conf = torch.tensor([random.uniform(0.4, 0.9)], device=Config.DEVICE)
        target_sl = torch.tensor([random.uniform(*Config.SL_MULTIPLIER_RANGE)], device=Config.DEVICE)
        target_tp = torch.tensor([random.uniform(*Config.TP_MULTIPLIER_RANGE)], device=Config.DEVICE)
        conf_loss = nn.MSELoss()(confidence.unsqueeze(0), target_conf)
        sl_loss = nn.MSELoss()(sl_mult.unsqueeze(0), target_sl)
        tp_loss = nn.MSELoss()(tp_mult.unsqueeze(0), target_tp)
        total_loss = conf_loss + 0.3 * (sl_loss + tp_loss)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
        learner.optimizer.step()
        symbol_str = f" | Symbol: {symbol}" if symbol else ""
        trade_logger.info(
            f"PASSIVE LEARNING{symbol_str} | Conf: {float(confidence):.2f}->{float(target_conf):.2f} | "
            f"SL: {float(sl_mult):.2f}->{float(target_sl):.2f} | "
            f"TP: {float(tp_mult):.2f}->{float(target_tp):.2f} | "
            f"Loss: {total_loss.item():.4f}"
        )
    except Exception as e:
        trade_logger.error(f"Passive learning failed: {e}")

def detect_market_regime(symbol):
    df = fetch_latest_data(symbol)
    if df is None:
        return "neutral"
    df = calculate_indicators(df, symbol, include_news=False)
    if df is None:
        return "neutral"
    last_atr = df['ATR_14'].iloc[-1]
    bb_width = df['BB_width'].iloc[-1]
    adx = df['ADX_14'].iloc[-1]
    volatility = df['volatility'].iloc[-1]
    regime = (
        "trending" if (adx > 25 and bb_width > 2*last_atr) else
        "volatile" if (volatility > df['volatility'].rolling(50).mean().iloc[-1]) else
        "range-bound"
    )
    logger.info(
        f"Regime Detection | {symbol} | "
        f"ATR: {last_atr:.4f} | BB Width: {bb_width:.4f} | "
        f"ADX: {adx:.1f} | Volatility: {volatility:.4f} -> {regime}"
    )
    return regime

def analyze_optimal_close(position):
    df = fetch_latest_data(position.symbol)
    if df is None:
        logger.warning(f"Optimal Close Analysis Failed | {position.symbol} | No Data")
        return "unknown"
    try:
        exit_idx = df.index.get_indexer([position.time_update], method='nearest')[0]
        future_bars = df.iloc[exit_idx+1 : exit_idx+13]  # Next 12 bars (1h)
        if position.type == mt5.ORDER_TYPE_BUY:
            missed_pips = (future_bars['high'].max() - position.price_current) / Config.PIP_SIZES[position.symbol]
        else:
            missed_pips = (position.price_current - future_bars['low'].min()) / Config.PIP_SIZES[position.symbol]
        result = (
            "early_close" if missed_pips > 20 else
            "late_close" if position.profit < 0 and missed_pips < -10 else
            "optimal"
        )
        trade_logger.info(
            f"TRADE ANALYSIS | Ticket: {position.ticket} | Symbol: {position.symbol} | "
            f"Type: {'BUY' if position.type == mt5.ORDER_TYPE_BUY else 'SELL'} | "
            f"Closed PnL: ${position.profit:.2f} | "
            f"Missed Pips: {missed_pips:.1f} | "
            f"Verdict: {result}"
        )
        return result
    except Exception as e:
        trade_logger.error(f"Close Analysis Error | {position.symbol}: {str(e)}")
        return "unknown"

def update_model_with_trade(position, reason):
    trade_logger.info(
        f"LEARNING STARTED | Ticket: {position.ticket} | Symbol: {position.symbol} | "
        f"Reason: {reason} | PnL: ${position.profit:.2f}"
    )
    snapshot = load_entry_snapshot(position.ticket)
    if snapshot is None:
        trade_logger.warning(f"No Snapshot for #{position.ticket} - skipping learning")
        return
    try:
        features = np.array(snapshot["features"])
        outcome = 1 if position.profit > 0 else 0
        current_regime = detect_market_regime(position.symbol)
        close_quality = analyze_optimal_close(position)
        risk_amount = Config.INITIAL_BALANCE * Config.MIN_RISK
        pnl_weight = min(2.0, abs(position.profit) / risk_amount)
        base_confidence = snapshot.get('confidence', 0.5)
        
        trade_logger.info(
            f"LEARNING DATA | Ticket: {position.ticket} | "
            f"Outcome: {'WIN' if outcome else 'LOSS'} | "
            f"Base Confidence: {base_confidence:.2f} | "
            f"Regime: {current_regime} | Close Quality: {close_quality} | "
            f"PnL Weight: {pnl_weight:.2f}"
        )

        # Dynamic confidence adjustment
        if outcome:  # Winning trade
            conf_adjust = 0.15 * pnl_weight
            conf_target = min(0.95, base_confidence + conf_adjust)
            if current_regime == "trending":
                conf_target = min(0.98, conf_target + 0.05)
        else:  # Losing trade
            conf_adjust = -0.2 * pnl_weight
            conf_target = max(0.05, base_confidence + conf_adjust)
        
        # SL/TP adjustments
        sl_mult_target = None
        tp_mult_target = None
        if outcome and close_quality == "early_close":
            sl_adjust = 0.1 * pnl_weight
            sl_mult_target = min(Config.SL_MULTIPLIER_RANGE[1], snapshot.get('sl_mult', 2.0) * (1.0 + sl_adjust))
        elif not outcome and close_quality == "late_close":
            tp_adjust = -0.15 * pnl_weight
            tp_mult_target = max(Config.TP_MULTIPLIER_RANGE[0], snapshot.get('tp_mult', 3.0) * (1.0 + tp_adjust))
        
        # Store trade in memory
        priority = abs(outcome - base_confidence) * (1.0 + pnl_weight)
        trade_memory.append({
            'features': features,
            'outcome': outcome,
            'regime': current_regime,
            'timestamp': datetime.utcnow().isoformat(),
            'confidence': base_confidence,
            'priority': priority,
            'pnl_weight': pnl_weight
        })
        
        # Update model
        learner.optimizer.zero_grad()
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(Config.DEVICE)
        actions, _ = model(features_tensor)
        signal, sl_mult, tp_mult, confidence = actions[0, 0]
        
        # Calculate losses
        conf_loss = nn.MSELoss()(confidence, torch.tensor([conf_target], device=Config.DEVICE))
        sl_loss = 0
        if sl_mult_target is not None:
            sl_loss = 0.5 * nn.MSELoss()(sl_mult, torch.tensor([sl_mult_target], device=Config.DEVICE))
        tp_loss = 0
        if tp_mult_target is not None:
            tp_loss = 0.5 * nn.MSELoss()(tp_mult, torch.tensor([tp_mult_target], device=Config.DEVICE))
        ewc_loss = learner.elastic_weight_consolidation()
        total_loss = conf_loss + sl_loss + tp_loss + ewc_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
        learner.optimizer.step()
        
        # Update EWC parameters
        learner.fisher_matrix = learner.compute_fisher_matrix()
        learner.previous_params = {n: p.clone() for n, p in model.named_parameters()}
        
        # Log learning update
        trade_logger.info(
            f"LEARNING UPDATE | Ticket: {position.ticket} | Symbol: {position.symbol} | "
            f"Type: {'BUY' if position.type == mt5.ORDER_TYPE_BUY else 'SELL'} | "
            f"Outcome: {'WIN' if outcome else 'LOSS'} | "
            f"Confidence: {base_confidence:.2f}->{conf_target:.2f} | "
            f"SL Mult: {float(sl_mult):.2f}" + (f"->{sl_mult_target:.2f}" if sl_mult_target else "") + " | "
            f"TP Mult: {float(tp_mult):.2f}" + (f"->{tp_mult_target:.2f}" if tp_mult_target else "") + " | "
            f"Regime: {current_regime} | Close Quality: {close_quality} | "
            f"Total Loss: {total_loss.item():.4f} | EWC Loss: {ewc_loss.item():.4f}"
        )
    except Exception as e:
        trade_logger.error(f"Live Learning Failed | {position.ticket}: {str(e)}")

def overnight_batch_learn():
    if len(trade_memory) < 100:
        trade_logger.info("Skipping batch learning - insufficient trades in memory")
        return
    learner.adapt_architecture()
    batch, weights = learner.prioritized_experience_replay(batch_size=64)
    features = torch.stack([torch.FloatTensor(x['features']) for x in batch]).to(Config.DEVICE)
    outcomes = torch.FloatTensor([x['outcome'] for x in batch]).to(Config.DEVICE)
    weights = weights.to(Config.DEVICE)
    learner.optimizer.zero_grad()
    actions, _ = model(features)
    confidences = actions[:, 3]
    loss = nn.BCELoss(weight=weights)(confidences, outcomes)
    loss.backward()
    learner.optimizer.step()
    learner.scheduler.step(loss)
    trade_logger.info(
        f"BATCH LEARNING | Trades: {len(batch)} | Loss: {loss.item():.4f} | "
        f"LR: {learner.optimizer.param_groups[0]['lr']:.2e}"
    )

# ================ TRADE EXECUTION ================
def calculate_position_size(symbol, risk_frac, sl_pips, entry_price):
    if symbol == "XAUUSD":
        pip_value = 10.0
    elif "JPY" in symbol:
        pip_value = 1000.0 / entry_price
    else:
        pip_value = 10.0 / entry_price
    risk_amount = Config.INITIAL_BALANCE * risk_frac
    lot_size = risk_amount / (sl_pips * pip_value)
    lot_size = float(lot_size)
    lot_size = round(lot_size / 0.01) * 0.01
    return max(0.01, min(lot_size, 100.0))

def check_correlated_positions(symbol, direction):
    positions = mt5.positions_get()
    if positions is None:
        return True
    for pos in positions:
        if pos.symbol == symbol:
            continue
        correlation = Config.CORRELATION_MATRIX[symbol].get(pos.symbol, 0)
        if abs(correlation) > 0.7:
            if (direction > 0.5 and pos.type == mt5.ORDER_TYPE_BUY and correlation > 0) or \
               (direction <= 0.5 and pos.type == mt5.ORDER_TYPE_SELL and correlation > 0) or \
               (direction > 0.5 and pos.type == mt5.ORDER_TYPE_SELL and correlation < 0) or \
               (direction <= 0.5 and pos.type == mt5.ORDER_TYPE_BUY and correlation < 0):
                trade_logger.info(f"Correlation Block | Symbol: {symbol} | Correlated: {pos.symbol} ({correlation:.2f})")
                return False
    return True

def execute_trade(symbol, signal, confidence, sl_mult, tp_mult):
    if confidence < Config.CONFIDENCE_THRESHOLD:
        trade_logger.info(f"Confidence Block | Symbol: {symbol} | Confidence: {confidence:.2f} < {Config.CONFIDENCE_THRESHOLD}")
        return None
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        trade_logger.error(f"Position Check Failed | Symbol: {symbol} | Error: {mt5.last_error()}")
        return None
    if len(positions) > 0:
        trade_logger.info(f"Position Block | Symbol: {symbol} | Existing positions: {len(positions)}")
        return None
    if not check_correlated_positions(symbol, signal):
        return None
        
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        trade_logger.error(f"Tick Data Failed | Symbol: {symbol} | Error: {mt5.last_error()}")
        return None
    price = tick.ask if signal > 0.5 else tick.bid
    direction = "BUY" if signal > 0.5 else "SELL"
    
    base_atr_pips = get_h1_atr(symbol)
    if base_atr_pips is None:
        trade_logger.error(f"ATR Failed | Symbol: {symbol}")
        return None
        
    raw_sl_mult = sl_mult
    raw_tp_mult = tp_mult
    current_regime = detect_market_regime(symbol)
    current_hour = datetime.utcnow().hour
    
    if current_regime == "range-bound":
        tp_mult = min(tp_mult, 3.0)
        sl_mult = max(sl_mult, 1.5)
    elif current_regime == "volatile":
        tp_mult = min(tp_mult, 4.0)
    if 0 <= current_hour < 8:
        tp_mult = max(1.8, tp_mult * 0.8)
        sl_mult = max(1.2, sl_mult * 0.9)
        
    pip = Config.PIP_SIZES.get(symbol, 0.0001)
    sl_pips = base_atr_pips * float(sl_mult)
    tp_pips = base_atr_pips * float(tp_mult)
    
    min_stop = Config.MIN_STOP_PIPS.get(symbol, 10)
    if sl_pips < min_stop:
        trade_logger.warning(f"SL Adjustment | Symbol: {symbol} | From: {sl_pips:.1f} to Min: {min_stop}")
        sl_pips = min_stop
        
    if tp_pips / sl_pips < Config.MIN_RR_RATIO:
        new_tp = sl_pips * Config.MIN_RR_RATIO
        trade_logger.warning(f"RR Adjustment | Symbol: {symbol} | Ratio: {tp_pips/sl_pips:.1f}->{Config.MIN_RR_RATIO} | TP: {tp_pips:.1f}->{new_tp:.1f}")
        tp_pips = new_tp
        
    risk_frac = Config.MIN_RISK + confidence * (Config.MAX_RISK - Config.MIN_RISK)
    lot_size = calculate_position_size(symbol, risk_frac, sl_pips, price)
    
    if direction == "BUY":
        sl_price = price - sl_pips * pip
        tp_price = price + tp_pips * pip
    else:
        sl_price = price + sl_pips * pip
        tp_price = price - tp_pips * pip
        
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        trade_logger.error(f"Symbol Info Failed | Symbol: {symbol}")
        return None
    digits = symbol_info.digits
    sl_price = round(sl_price, digits)
    tp_price = round(tp_price, digits)
    filling_mode = mt5.ORDER_FILLING_IOC
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL,
        "price": price,
        "sl": sl_price,
        "tp": tp_price,
        "deviation": 10,
        "magic": 123456,
        "comment": f"AI (Conf: {confidence:.2f})",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": filling_mode,
    }
    
    result = mt5.order_send(request)
    if result is None:
        trade_logger.error(f"Order Send Failed | Symbol: {symbol} | Error: {mt5.last_error()}")
        return None
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        trade_logger.error(f"Order Rejected | Symbol: {symbol} | Retcode: {result.retcode} | Comment: {result.comment}")
        return None
        
    trade_logger.info(
        f"TRADE OPENED | Ticket: {result.order} | Symbol: {symbol} | "
        f"Type: {direction} | Lots: {lot_size:.2f} | Price: {price:.5f} | "
        f"SL: {sl_price:.5f} | TP: {tp_price:.5f} | Confidence: {confidence:.2f} | "
        f"Risk: {risk_frac*100:.1f}% | H1 ATR: {base_atr_pips:.1f} pips | "
        f"SL Multiplier: {sl_mult:.2f}x | TP Multiplier: {tp_mult:.2f}x | "
        f"SL Pips: {sl_pips:.1f} | TP Pips: {tp_pips:.1f} | RR Ratio: {tp_pips/sl_pips:.2f} | "
        f"Regime: {current_regime} | Hour: {current_hour}"
    )
    
    try:
        df = fetch_latest_data(symbol)
        if df is not None:
            df = calculate_indicators(df, symbol)
            if df is not None:
                features = df[Config.FEATURE_COLS].values[-Config.SEQ_LEN:]
                save_entry_snapshot(result.order, symbol, features, raw_sl_mult, raw_tp_mult, confidence)
                trade_logger.info(f"Snapshot Saved | Ticket: {result.order}")
    except Exception as e:
        trade_logger.error(f"Snapshot Failed | Ticket: {result.order} | Error: {e}")
    
    positions = mt5.positions_get(symbol=symbol)
    if positions and len(positions) > 0:
        position = positions[0]
        trade_outcomes.append(1 if position.profit > 0 else 0)
    else:
        trade_outcomes.append(0)
    if len(trade_outcomes) % 10 == 0:
        auto_adjust_thresholds()
    return result

def close_trade(ticket, reason):
    positions = mt5.positions_get(ticket=ticket)
    if positions is None:
        trade_logger.error(f"Position Fetch Failed | Ticket: {ticket} | Error: {mt5.last_error()}")
        return
    if len(positions) == 0:
        trade_logger.error(f"Position Not Found | Ticket: {ticket}")
        return
    position = positions[0]
    close_price = mt5.symbol_info_tick(position.symbol).bid if position.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(position.symbol).ask
    
    close_request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": position.symbol,
        "volume": position.volume,
        "type": mt5.ORDER_TYPE_BUY if position.type == mt5.ORDER_TYPE_SELL else mt5.ORDER_TYPE_SELL,
        "position": position.ticket,
        "price": close_price,
        "deviation": 10,
        "magic": position.magic,
        "comment": f"Closed: {reason}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    result = mt5.order_send(close_request)
    if result is None:
        trade_logger.error(f"Close Failed | Ticket: {ticket} | Error: {mt5.last_error()}")
        return
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        trade_logger.error(f"Close Rejected | Ticket: {ticket} | Retcode: {result.retcode} | Comment: {result.comment}")
        return
        
    trade_logger.info(
        f"TRADE CLOSED | Ticket: {position.ticket} | Symbol: {position.symbol} | "
        f"Type: {'BUY' if position.type == mt5.ORDER_TYPE_BUY else 'SELL'} | "
        f"Entry: {position.price_open:.5f} | Exit: {close_price:.5f} | "
        f"Lots: {position.volume:.2f} | P/L: ${position.profit:.2f} | "
        f"Reason: {reason}"
    )
    
    update_model_with_trade(position, reason)

# ================ TRADE MEMORY SYSTEM ================
def save_entry_snapshot(ticket, symbol, features, sl_mult, tp_mult, confidence):
    try:
        os.makedirs(Config.TRADE_MEMORY_DIR, exist_ok=True)
        file_path = f"{Config.TRADE_MEMORY_DIR}/{ticket}.json"
        snapshot = {
            "timestamp": datetime.utcnow().isoformat(),
            "symbol": symbol,
            "features": features.tolist(),
            "sl_mult": float(sl_mult),
            "tp_mult": float(tp_mult),
            "confidence": float(confidence)
        }
        with open(file_path, 'w') as f:
            json.dump(snapshot, f)
    except Exception as e:
        trade_logger.error(f"Snapshot Save Failed | Ticket: {ticket} | Error: {str(e)}")

def load_entry_snapshot(ticket):
    try:
        file_path = f"{Config.TRADE_MEMORY_DIR}/{ticket}.json"
        if not os.path.exists(file_path):
            return None
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        trade_logger.error(f"Snapshot Load Failed | Ticket: {ticket} | Error: {str(e)}")
        return None

def cleanup_old_snapshots():
    try:
        now = time.time()
        deleted = 0
        for filename in os.listdir(Config.TRADE_MEMORY_DIR):
            file_path = os.path.join(Config.TRADE_MEMORY_DIR, filename)
            if os.path.isfile(file_path):
                file_age = now - os.path.getmtime(file_path)
                if file_age > 5184000:
                    os.remove(file_path)
                    deleted += 1
        if deleted > 0:
            trade_logger.info(f"Snapshot Cleanup | Deleted: {deleted} files")
    except Exception as e:
        trade_logger.error(f"Cleanup Failed | Error: {str(e)}")

# ================ TRAILING STOP LOSS ================
def update_trailing_stops():
    if not is_market_open():
        return
    trade_logger.info("Trailing Stop Check Started")
    positions = mt5.positions_get()
    if positions is None:
        trade_logger.error(f"Position Fetch Failed | Error: {mt5.last_error()}")
        return
    for position in positions:
        try:
            symbol = position.symbol
            current_price = mt5.symbol_info_tick(symbol)
            if current_price is None:
                trade_logger.error(f"Price Check Failed | Symbol: {symbol}")
                continue
            if position.type == mt5.ORDER_TYPE_BUY:
                current_price = current_price.bid
            else:
                current_price = current_price.ask
            pip_size = Config.PIP_SIZES.get(symbol, 0.0001)
            entry_price = position.price_open
            if position.type == mt5.ORDER_TYPE_BUY:
                profit_pips = (current_price - entry_price) / pip_size
            else:
                profit_pips = (entry_price - current_price) / pip_size
                
            base_atr_pips = get_h1_atr(symbol)
            if base_atr_pips is None:
                trade_logger.info(f"ATR Failed | Symbol: {symbol} - skipping trailing")
                continue
            current_atr = base_atr_pips * pip_size
            
            if profit_pips < (base_atr_pips * 1.5):
                trade_logger.info(f"Trailing Inactive | Ticket: {position.ticket} | Profit: {profit_pips:.1f}pips < {base_atr_pips*1.5:.1f}pips")
                continue
                
            if position.type == mt5.ORDER_TYPE_BUY:
                new_sl = current_price - (current_atr * Config.TRAILING_ATR_MULTIPLIER)
                if new_sl > position.sl:
                    trade_logger.info(
                        f"Trailing Update | Ticket: {position.ticket} | Symbol: {symbol} | "
                        f"Type: BUY | Old SL: {position.sl:.5f} | New SL: {new_sl:.5f} | "
                        f"Current Price: {current_price:.5f} | Profit Pips: {profit_pips:.1f} | "
                        f"H1 ATR: {base_atr_pips:.1f}pips"
                    )
                    update_position_sl(position, new_sl)
                else:
                    trade_logger.info(f"Trailing Unchanged | Ticket: {position.ticket} | New SL: {new_sl:.5f} <= Current SL: {position.sl:.5f}")
            else:
                new_sl = current_price + (current_atr * Config.TRAILING_ATR_MULTIPLIER)
                if new_sl < position.sl or position.sl == 0:
                    trade_logger.info(
                        f"Trailing Update | Ticket: {position.ticket} | Symbol: {symbol} | "
                        f"Type: SELL | Old SL: {position.sl:.5f} | New SL: {new_sl:.5f} | "
                        f"Current Price: {current_price:.5f} | Profit Pips: {profit_pips:.1f} | "
                        f"H1 ATR: {base_atr_pips:.1f}pips"
                    )
                    update_position_sl(position, new_sl)
                else:
                    trade_logger.info(f"Trailing Unchanged | Ticket: {position.ticket} | New SL: {new_sl:.5f} >= Current SL: {position.sl:.5f}")
        except Exception as e:
            trade_logger.error(f"Trailing Error | Ticket: {position.ticket} | Error: {str(e)}")
            continue
    trade_logger.info("Trailing Stop Check Completed")

def update_position_sl(position, new_sl):
    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "position": position.ticket,
        "symbol": position.symbol,
        "sl": new_sl,
        "tp": position.tp,
    }
    result = mt5.order_send(request)
    if result is None:
        trade_logger.error(f"SL Update Failed | Ticket: {position.ticket} | Error: {mt5.last_error()}")
        return
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        trade_logger.error(f"SL Update Rejected | Ticket: {position.ticket} | Retcode: {result.retcode} | Comment: {result.comment}")
    else:
        trade_logger.info(
            f"SL MODIFIED | Ticket: {position.ticket} | Symbol: {position.symbol} | "
            f"New SL: {new_sl:.5f}"
        )

# ================ MODEL SAVING ================
def save_model():
    try:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': learner.optimizer.state_dict(),
            'fisher_matrix': learner.fisher_matrix,
            'previous_params': learner.previous_params
        }, Config.MODEL_BACKUP)
        temp_path = f"{Config.MODEL_FILE}.tmp"
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': learner.optimizer.state_dict(),
            'fisher_matrix': learner.fisher_matrix,
            'previous_params': learner.previous_params
        }, temp_path)
        os.replace(temp_path, Config.MODEL_FILE)
        trade_logger.info(f"Model Saved | Path: {Config.MODEL_FILE}")
    except Exception as e:
        trade_logger.error(f"Model Save Failed | Error: {e}")

def check_balance_limit():
    account = mt5.account_info()
    if account is None:
        trade_logger.error(f"Account Check Failed | Error: {mt5.last_error()}")
        return
    balance = account.balance
    if balance <= Config.BALANCE_LIMIT:
        trade_logger.critical(f"BALANCE LIMIT HIT | Balance: ${balance:.2f} <= Limit: ${Config.BALANCE_LIMIT:.2f}")
        positions = mt5.positions_get()
        if positions is None:
            trade_logger.error(f"Position Fetch Failed | Error: {mt5.last_error()}")
        elif positions:
            for position in positions:
                close_trade(position.ticket, "BALANCE_LIMIT")
        save_model()
        mt5.shutdown()
        exit()

# ================ AUTO-CLOSURE LOGGING ================
def log_auto_closures():
    now = datetime.utcnow()
    deals = mt5.history_deals_get(now - timedelta(minutes=5), now)
    if deals is None:
        return
    for deal in deals:
        if deal.entry in (mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_OUT_BY):
            reason = "SL" if deal.profit < 0 else "TP"
            trade_logger.info(
                f"AUTO-CLOSED | Ticket: {deal.position_id} | "
                f"Symbol: {deal.symbol} | Reason: {reason} | "
                f"PnL: ${deal.profit:.2f}"
            )

# ================ MAIN LOOP ================
if __name__ == "__main__":
    trade_logger.info("=============== FOREX AI STARTED ===============")
    initialize_mt5()
    initialize_model()
    news_last_checked = time.time()
    trailing_last_updated = time.time()
    last_bar_time = {}
    last_save_time = time.time()
    last_cleanup = time.time()
    last_passive_learn = time.time()
    last_closure_check = time.time()
    os.makedirs(Config.TRADE_MEMORY_DIR, exist_ok=True)
    os.makedirs(Config.TRADE_LOG_DIR, exist_ok=True)
    
    try:
        while True:
            current_time = time.time()
            check_balance_limit()
            
            # Periodic tasks
            if current_time - last_closure_check > 60:
                log_auto_closures()
                last_closure_check = current_time
            if current_time - last_save_time > 21600:
                save_model()
                last_save_time = current_time
            if current_time - last_cleanup > 86400:
                cleanup_old_snapshots()
                last_cleanup = current_time
            if current_time - news_last_checked > Config.NEWS_CHECK_INTERVAL:
                if is_market_open():
                    check_recent_news()
                news_last_checked = current_time
            if current_time - trailing_last_updated > 300:
                if is_market_open():
                    update_trailing_stops()
                trailing_last_updated = current_time
            
            # Trading logic
            if is_market_open():
                for symbol in Config.SYMBOLS:
                    rates = mt5.copy_rates_from_pos(symbol, Config.BAR_TIMEFRAME, 0, Config.INITIAL_DATA_WINDOW)
                    if rates is None or len(rates) == 0:
                        continue
                    current_bar_time_val = rates[0]['time']
                    bars_since_last_check = 0
                    if symbol in last_bar_time:
                        bars_since_last_check = (current_bar_time_val - last_bar_time[symbol]) // 300
                    else:
                        bars_since_last_check = Config.MEDIUM_CONFIDENCE_INTERVAL
                    last_bar_time[symbol] = current_bar_time_val
                    
                    should_check = (
                        (bars_since_last_check >= Config.HIGH_CONFIDENCE_INTERVAL) or
                        (bars_since_last_check >= Config.MEDIUM_CONFIDENCE_INTERVAL)
                    )
                    if not should_check:
                        continue
                    
                    df = fetch_latest_data(symbol)
                    if df is None:
                        continue
                    df = calculate_indicators(df, symbol)
                    if df is None:
                        continue
                    features = df[Config.FEATURE_COLS].values[-Config.SEQ_LEN:]
                    
                    with torch.no_grad():
                        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(Config.DEVICE)
                        actions, _ = model(features_tensor)
                        signal, sl_mult, tp_mult, confidence = actions[0, 0]
                        signal = float(signal)
                        sl_mult = float(sl_mult)
                        tp_mult = float(tp_mult)
                        confidence = float(confidence)
                    
                    trade_logger.info(
                        f"SIGNAL CHECK | Symbol: {symbol} | {'BUY' if signal > 0.5 else 'SELL'} | "
                        f"Confidence: {confidence:.2f} | SL Mult: {sl_mult:.2f} | TP Mult: {tp_mult:.2f} | "
                        f"Bars since last check: {bars_since_last_check}"
                    )
                    
                    if current_time - last_passive_learn > Config.PASSIVE_LEARN_INTERVAL:
                        passive_learn(features_tensor, symbol)
                        last_passive_learn = current_time
                    
                    if (bars_since_last_check >= Config.HIGH_CONFIDENCE_INTERVAL and
                        confidence >= Config.HIGH_CONFIDENCE_THRESHOLD) or (
                        bars_since_last_check >= Config.MEDIUM_CONFIDENCE_INTERVAL and
                        confidence >= Config.CONFIDENCE_THRESHOLD):
                        trade_result = execute_trade(symbol, signal, confidence, sl_mult, tp_mult)
            
            # Overnight learning
            if datetime.utcnow().hour == 2 and not is_market_open():
                overnight_batch_learn()
                
            time.sleep(1)
    except KeyboardInterrupt:
        trade_logger.info("Graceful Shutdown Initiated")
        save_model()
        mt5.shutdown()
    except Exception as e:
        trade_logger.critical(f"FATAL ERROR: {e}", exc_info=True)
        save_model()
        mt5.shutdown()