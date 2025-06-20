import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import requests
from datetime import datetime, timedelta
import pytz
import os
import joblib
import ta
from tensorflow.keras.models import load_model

# ========== CONFIGURATION ==========
SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "XAUUSD"]
TIMEFRAME = mt5.TIMEFRAME_M5
FIXED_CONFIDENCE_THRESHOLD = 0.65
MIN_RISK = 0.005
MAX_RISK = 0.02
MIN_LOT_SIZE = 0.01
MAX_LOT_SIZE = 20.0
COMMISSION_PIPS = 0.02
SLIPPAGE_RATIO = 0.3
MIN_STOP_DISTANCE_PIPS = 10
DRAWDOWN_LIMIT_PCT = 0.25

# Multiplier ranges
MIN_SL_MULT = 1.0
MAX_SL_MULT = 5.0
MIN_TP_MULT = 1.0
MAX_TP_MULT = 5.0

# Trade limits
MAX_OPEN_TRADES = 3

# News and model paths
EOD_API_KEY = "684d76d81e83b0.88582643"
NEWS_CHECK_INTERVAL = 45
MODEL_DIR = "C:/FPFX/model/"
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

# ========== MT5 CONNECTION ==========
def connect_mt5():
    print("\n[CONNECTION] Initializing MT5 connection...")
    if not mt5.initialize():
        print("[ERROR] Failed to initialize MT5")
        quit()
    
    account = 52380678
    password = "OlN&E$83kbSNYX"
    server = "ICMarketsSC-Demo"
    
    authorized = mt5.login(account, password, server)
    if not authorized:
        print("[ERROR] Failed to connect to MT5 account")
        mt5.shutdown()
        quit()
    
    balance = mt5.account_info().balance
    print(f"[SUCCESS] Connected to MT5 Demo Account: {account}")
    print(f"[BALANCE] Detected Account Balance: ${balance:.2f}")
    return balance

# ========== DATA PROCESSING ==========
def get_pip_size(symbol):
    if symbol == "XAUUSD":
        return 0.01  # Gold has 2 decimal places but pip size is 0.01
    return 0.01 if "JPY" in symbol else 0.0001

def fetch_historical_data(symbol, bars=300):
    print(f"[DATA] Fetching {bars} bars of {symbol}...")
    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 0, bars)
    if rates is None:
        print(f"[ERROR] Failed to fetch data for {symbol}")
        return None
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    
    # Technical indicators
    df['RSI_14'] = ta.momentum.rsi(df['close'], window=14)
    df['BB_%B'] = ta.volatility.bollinger_pband(df['close'], window=20, window_dev=2)
    df['ATR_14'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=14, smooth_window=3)
    df['STOCH_%K'] = stoch.stoch()
    df['STOCH_%D'] = stoch.stoch_signal()
    macd = ta.trend.MACD(df['close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    kc = ta.volatility.KeltnerChannel(df['high'], df['low'], df['close'], window=20, window_atr=10)
    df['KC_upper'] = kc.keltner_channel_hband()
    df['KC_middle'] = kc.keltner_channel_mband()
    df['KC_lower'] = kc.keltner_channel_lband()
    df['SMA_50'] = ta.trend.sma_indicator(df['close'], window=50)
    df['SMA_200'] = ta.trend.sma_indicator(df['close'], window=200)
    df['ADX_14'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
    psar = ta.trend.PSARIndicator(df['high'], df['low'], df['close'], step=0.02, max_step=0.2)
    df['PSAR'] = psar.psar()
    df['TRIX_15'] = ta.trend.trix(df['close'], window=15)
    df['VWMA_20'] = (df['close'] * df['tick_volume']).rolling(20).sum() / df['tick_volume'].rolling(20).sum()
    df['volume'] = df['tick_volume']
    
    # Regime calculation
    hist = df['MACD'] - df['MACD_signal']
    adx_med = df['ADX_14'].median()
    df['Regime0'] = ((df['ADX_14'] <= adx_med) & (hist < 0)).astype(int)
    df['Regime1'] = ((df['ADX_14'] <= adx_med) & (hist >= 0)).astype(int)
    df['Regime2'] = ((df['ADX_14'] > adx_med) & (hist < 0)).astype(int)
    df['Regime3'] = ((df['ADX_14'] > adx_med) & (hist >= 0)).astype(int)
    
    df.dropna(inplace=True)
    return df

def prepare_features(df, news_count=0, avg_sentiment=0.0):
    if df is None or df.empty:
        return None
    
    # Match the exact feature order used in training
    features = df[[
        'open','high','low','close','volume',
        'RSI_14','BB_%B','ATR_14','STOCH_%K','STOCH_%D',
        'MACD','MACD_signal','KC_upper','KC_middle','KC_lower',
        'SMA_50','ADX_14','PSAR','SMA_200','TRIX_15',
        'VWMA_20','Regime0','Regime1','Regime2','Regime3'
    ]].iloc[-1].values
    
    # Append news features
    features = np.append(features, [news_count, avg_sentiment])
    return features

# ========== NEWS SENTIMENT ==========
def fetch_news_sentiment(symbol):
    ticker = f"{symbol}.FOREX"
    now = datetime.now(pytz.utc)
    params = {
        "s": ticker,
        "offset": 0,
        "limit": 100,
        "api_token": EOD_API_KEY,
        "fmt": "json"
    }
    url = "https://eodhd.com/api/news"
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"[NEWS ERROR] Failed to fetch news: {str(e)}")
        return 0, 0.0
    
    recent = []
    for item in data:
        pub = item.get("published_at") or item.get("date") or item.get("datetime")
        if not pub:
            continue
        
        try:
            pub_dt = datetime.fromisoformat(pub.replace('Z', '+00:00'))
        except ValueError:
            continue
            
        age = (now - pub_dt).total_seconds()
        if age <= 3600:  # 1-hour window
            pol = item.get("sentiment", {}).get("polarity", 0.0)
            weight = np.exp(-age / 120)
            recent.append((pol, weight))
    
    if not recent:
        return 0, 0.0
    
    total_w = sum(w for _, w in recent)
    avg_weighted = sum(p * w for p, w in recent) / total_w
    return len(recent), float(avg_weighted)

# ========== TRADE EXECUTION ==========
def get_current_price(symbol):
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return None
    return (tick.bid + tick.ask)/2

def calculate_position_size(balance, risk_pct, stop_pips, symbol, price):
    if symbol == "XAUUSD":
        pip_value = 1.0
    elif "JPY" in symbol and "USD" in symbol:
        pip_value = 1000 / price
    else:
        pip_value = 10.0
    
    risk_amount = balance * risk_pct
    lot_size = risk_amount / (stop_pips * pip_value)
    
    # Apply broker lot step constraints
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info:
        step = symbol_info.volume_step
        min_lot = symbol_info.volume_min
        max_lot = symbol_info.volume_max
        
        # Round to nearest step
        if step > 0:
            lot_size = round(lot_size / step) * step
        
        lot_size = max(min(lot_size, max_lot), min_lot)
    else:
        lot_size = max(min(lot_size, MAX_LOT_SIZE), MIN_LOT_SIZE)
    
    return lot_size

def get_open_positions():
    positions = mt5.positions_get()
    if positions is None:
        return []
    return positions

def execute_trade(symbol, action, balance, hist_data):
    # Check max open trades and existing position for symbol
    positions = get_open_positions()
    total_positions = len(positions)
    symbol_positions = [p for p in positions if p.symbol == symbol]
    
    if total_positions >= MAX_OPEN_TRADES:
        print(f"[SKIP] Max open trades reached ({MAX_OPEN_TRADES}) - skipping")
        return
    
    if symbol_positions:
        print(f"[SKIP] {symbol} already has open position - skipping")
        return
    
    signal, sl_mult, tp_mult, sent_exit_thresh, confidence = action
    print(f"\n[TRADE SIGNAL] {symbol} Analysis:")
    print(f" Signal: {'BUY' if signal >= 0.5 else 'SELL'} ({signal:.2f})")
    print(f" SL Multiplier: {sl_mult:.2f}x ATR")
    print(f" TP Multiplier: {tp_mult:.2f}x ATR")
    print(f" Sentiment Threshold: {sent_exit_thresh:.2f}")
    print(f" Confidence: {confidence:.2f}")
    
    if confidence < FIXED_CONFIDENCE_THRESHOLD:
        print(f"[DECISION] Confidence below threshold {FIXED_CONFIDENCE_THRESHOLD:.2f} - No trade")
        return
    
    pip_size = get_pip_size(symbol)
    price = get_current_price(symbol)
    if price is None:
        print(f"[ERROR] Failed to get current price for {symbol}")
        return
    
    # Calculate ATR in pips
    if symbol != "XAUUSD":
        atr_pips = (hist_data['ATR_14'].iloc[-1] * 10) / pip_size
    else:
        atr_pips = hist_data['ATR_14'].iloc[-1] / pip_size
    
    # Calculate stop distance
    stop_distance_pips = atr_pips * sl_mult
    stop_distance_pips = max(stop_distance_pips, MIN_STOP_DISTANCE_PIPS)
    
    # Calculate TP distance
    tp_distance_pips = atr_pips * tp_mult
    
    # Enforce positive risk-reward ratio
    if tp_distance_pips <= stop_distance_pips:
        print(f"[SKIP] Risk:Reward <= 1 ({tp_distance_pips:.1f}p TP <= {stop_distance_pips:.1f}p SL)")
        return
    
    # Calculate risk percentage
    risk_pct = MIN_RISK + confidence * (MAX_RISK - MIN_RISK)
    risk_pct = min(risk_pct, MAX_RISK)
    
    # Get symbol info for lot calculation
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"[ERROR] Could not get symbol info for {symbol}")
        return
    
    # Calculate position size with proper rounding
    lot_size = calculate_position_size(balance, risk_pct, stop_distance_pips, symbol, price)
    
    # Determine direction and prices
    direction = mt5.ORDER_TYPE_BUY if signal >= 0.5 else mt5.ORDER_TYPE_SELL
    
    if direction == mt5.ORDER_TYPE_BUY:
        sl_price = price - stop_distance_pips * pip_size
        tp_price = price + tp_distance_pips * pip_size
    else:
        sl_price = price + stop_distance_pips * pip_size
        tp_price = price - tp_distance_pips * pip_size
    
    # Validate prices
    if sl_price <= 0 or tp_price <= 0:
        print(f"[ERROR] Invalid price levels - SL: {sl_price:.5f}, TP: {tp_price:.5f}")
        return
    
    # Check broker restrictions
    point = symbol_info.point
    min_stop = symbol_info.trade_stops_level * point
    min_stop_pips = min_stop / pip_size
    
    if stop_distance_pips < min_stop_pips:
        print(f"[ERROR] Stop distance {stop_distance_pips:.1f} pips < broker minimum {min_stop_pips:.1f} pips")
        return
    
    print(f"[TRADE DETAILS] {symbol} {'BUY' if direction == mt5.ORDER_TYPE_BUY else 'SELL'}")
    print(f" Entry: {price:.5f}")
    print(f" SL: {sl_price:.5f} ({stop_distance_pips:.1f} pips)")
    print(f" TP: {tp_price:.5f} ({tp_distance_pips:.1f} pips)")
    print(f" Lots: {lot_size:.2f}")
    print(f" Risk: {risk_pct*100:.2f}% (${balance*risk_pct:.2f})")
    print(f" ATR: {atr_pips:.1f} pips (raw: {hist_data['ATR_14'].iloc[-1]:.6f})")
    
    # Get appropriate price based on direction
    if direction == mt5.ORDER_TYPE_BUY:
        order_price = mt5.symbol_info_tick(symbol).ask
    else:
        order_price = mt5.symbol_info_tick(symbol).bid
    
    # Immediate or Cancel filling mode
    filling_mode = mt5.ORDER_FILLING_IOC
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": direction,
        "price": order_price,
        "sl": sl_price,
        "tp": tp_price,
        "deviation": 20,
        "comment": "AI Trade",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": filling_mode
    }
    
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"[ORDER FAILED] {symbol} - Error: {result.comment}")
        print(f" Retcode: {result.retcode}")
        print(f" Requested lot size: {lot_size:.5f}")
    else:
        print(f"[ORDER EXECUTED] {symbol} - Ticket: {result.order}")

def check_exits(symbol, news_count, avg_sentiment):
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return
    
    for pos in positions:
        typ = pos.type
        if ((typ == mt5.ORDER_TYPE_BUY and avg_sentiment <= -0.5) or
            (typ == mt5.ORDER_TYPE_SELL and avg_sentiment >= 0.5)):
            close_position(pos)

def close_position(position):
    tick = mt5.symbol_info_tick(position.symbol)
    if tick is None:
        return
    
    price = tick.ask if position.type == mt5.ORDER_TYPE_BUY else tick.bid
    
    # Immediate or Cancel filling mode
    filling_mode = mt5.ORDER_FILLING_IOC
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "position": position.ticket,
        "symbol": position.symbol,
        "volume": position.volume,
        "type": mt5.ORDER_TYPE_BUY if position.type == mt5.ORDER_TYPE_SELL else mt5.ORDER_TYPE_SELL,
        "price": price,
        "deviation": 20,
        "comment": "AI Exit",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": filling_mode
    }
    
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"[EXIT FAILED] {position.symbol} - Error: {result.comment}")
    else:
        print(f"[POSITION CLOSED] {position.symbol} - P/L: ${position.profit:.2f}")

# ========== MAIN LOOP ==========
def main():
    initial_balance = connect_mt5()
    drawdown_limit = initial_balance * (1 - DRAWDOWN_LIMIT_PCT)
    print(f"\n[ACCOUNT] Starting balance: ${initial_balance:.2f}")
    print(f"[RISK] Daily drawdown limit: ${drawdown_limit:.2f}")
    
    # Load models
    try:
        actor = load_model(os.path.join(MODEL_DIR, "actor.h5"))
        scaler = joblib.load(SCALER_PATH)
    except Exception as e:
        print(f"[MODEL ERROR] Failed to load models: {str(e)}")
        mt5.shutdown()
        quit()
    
    news_cache = {s: {"count": 0, "sentiment": 0.0} for s in SYMBOLS}
    last_news_check = datetime.now() - timedelta(seconds=NEWS_CHECK_INTERVAL)
    
    while True:
        try:
            account_info = mt5.account_info()
            if account_info is None:
                print("[ERROR] Failed to get account info")
                time.sleep(10)
                continue
                
            current_balance = account_info.balance
            if current_balance <= drawdown_limit:
                print(f"\n[RISK LIMIT] Account reached drawdown limit: ${current_balance:.2f}")
                break
            
            print(f"\n[STATUS] Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Balance: ${current_balance:.2f}")
            
            # Check news sentiment periodically
            current_time = datetime.now()
            if (current_time - last_news_check).seconds >= NEWS_CHECK_INTERVAL:
                print("\n[NEWS] Updating sentiment data...")
                for symbol in SYMBOLS:
                    cnt, sent = fetch_news_sentiment(symbol)
                    news_cache[symbol] = {"count": cnt, "sentiment": sent}
                    print(f" {symbol}: {cnt} news items | Sentiment: {sent:.2f}")
                last_news_check = current_time
            
            # Get current positions
            positions = get_open_positions()
            total_positions = len(positions)
            print(f"[POSITIONS] Open trades: {total_positions}/{MAX_OPEN_TRADES}")
            
            # Check each symbol for trades
            for symbol in SYMBOLS:
                try:
                    print(f"\n[ANALYZING] {symbol}...")
                    
                    # Skip if symbol already has position
                    if any(p.symbol == symbol for p in positions):
                        print(f"[SKIP] {symbol} has existing position - skipping analysis")
                        continue
                    
                    # Skip if max positions reached
                    if total_positions >= MAX_OPEN_TRADES:
                        print(f"[SKIP] Max trades open ({MAX_OPEN_TRADES}) - skipping analysis")
                        continue
                    
                    # Fetch historical data
                    hist_data = fetch_historical_data(symbol)
                    if hist_data is None or hist_data.empty:
                        print(f"[ERROR] No data for {symbol} - skipping")
                        continue
                    
                    # Prepare features
                    raw_features = prepare_features(
                        hist_data,
                        news_cache[symbol]["count"],
                        news_cache[symbol]["sentiment"]
                    )
                    if raw_features is None:
                        print(f"[ERROR] Failed to prepare features for {symbol}")
                        continue
                    
                    # Scale features
                    features = scaler.transform([raw_features])[0]
                    
                    # One-hot encode symbol
                    oh = np.zeros(len(SYMBOLS))
                    oh[SYMBOLS.index(symbol)] = 1
                    
                    # Combine features
                    obs = np.concatenate([features, oh])
                    inp = obs.reshape(1, 1, -1)
                    
                    # Get prediction
                    action = actor.predict(inp, verbose=0)[0]
                    
                    # Clip to valid ranges
                    action[0] = np.clip(action[0], 0.0, 1.0)  # Signal
                    action[1] = np.clip(action[1], MIN_SL_MULT, MAX_SL_MULT)  # SL multiplier
                    action[2] = np.clip(action[2], MIN_TP_MULT, MAX_TP_MULT)  # TP multiplier
                    action[3] = np.clip(action[3], -0.1, 0.1)  # Sentiment threshold
                    action[4] = np.clip(action[4], 0.0, 1.0)  # Confidence
                    
                    # Print detailed 5D action vector
                    print(f"[5D ACTION VECTOR] Signal={action[0]:.3f}, SL_mult={action[1]:.3f}, "
                          f"TP_mult={action[2]:.3f}, Sent_thresh={action[3]:.3f}, Confidence={action[4]:.3f}")
                    
                    execute_trade(symbol, action, current_balance, hist_data)
                    check_exits(symbol, news_cache[symbol]["count"], news_cache[symbol]["sentiment"])
                
                except Exception as e:
                    print(f"[ERROR] Processing {symbol}: {str(e)}")
            
            # Sleep until next 5-minute candle
            now = datetime.now()
            next_run = (now + timedelta(minutes=5)).replace(second=0, microsecond=0)
            sleep_time = (next_run - now).total_seconds()
            print(f"\n[STATUS] Sleeping for {sleep_time:.1f} seconds...")
            time.sleep(max(sleep_time, 1))
        
        except Exception as e:
            print(f"\n[CRITICAL ERROR] {str(e)}")
            print("Waiting 60 seconds before retrying...")
            time.sleep(60)
    
    print("\n=== TRADING HALTED ===")
    mt5.shutdown()

if __name__ == "__main__":
    main()