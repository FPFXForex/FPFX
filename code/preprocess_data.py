import os
import pandas as pd
import pandas_ta as ta
import pytz
from datetime import datetime
import numpy as np

# Configuration
RAW_DATA_DIR = "C:/FPFX/Data"
PROCESSED_DIR = "C:/FPFX/Data/Processed"
SYMBOLS = ["AUDUSD", "EURUSD", "GBPUSD", "USDJPY", "USDCAD", "XAUUSD"]
START_DATE = "2023-06-12"
END_DATE = "2025-06-12"

def calculate_all_indicators(df):
    """Calculate all technical indicators with proper warm-up period"""
    df = df.copy()
    
    # Calculate basic price features
    df['range'] = df['high'] - df['low']
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    
    # Calculate indicators with proper warm-up periods
    min_calc_length = 200  # Enough for SMA200
    
    if len(df) > min_calc_length:
        # Volatility and momentum
        df['volatility'] = df['close'].pct_change().rolling(20).std()
        df['momentum'] = df['close'].pct_change(5)
        
        # SMAs
        df['SMA_50'] = ta.sma(df['close'], length=50)
        df['SMA_200'] = ta.sma(df['close'], length=200)
        
        # Oscillators
        df['RSI_14'] = ta.rsi(df['close'], length=14)
        stoch = ta.stoch(df['high'], df['low'], df['close'])
        df['STOCH_%K'] = stoch['STOCHk_14_3_3']
        df['STOCH_%D'] = stoch['STOCHd_14_3_3']
        
        # Trend indicators
        adx_data = ta.adx(df['high'], df['low'], df['close'], length=14)
        df['ADX_14'] = adx_data['ADX_14']
        df['DMP_14'] = adx_data['DMP_14']
        df['DMN_14'] = adx_data['DMN_14']
        
        # Bands
        bb = ta.bbands(df['close'], length=20)
        df['BB_%B'] = (df['close'] - bb['BBL_20_2.0']) / (bb['BBU_20_2.0'] - bb['BBL_20_2.0']).replace(0, 1e-10)
        df['BB_width'] = (bb['BBU_20_2.0'] - bb['BBL_20_2.0']) / bb['BBM_20_2.0']
        
        kc = ta.kc(df['high'], df['low'], df['close'])
        df['KC_upper'] = kc['KCUe_20_2']
        df['KC_middle'] = kc['KCBe_20_2']
        df['KC_lower'] = kc['KCLe_20_2']
        
        # MACD
        macd = ta.macd(df['close'])
        df['MACD_line'] = macd['MACD_12_26_9']
        df['MACD_signal'] = macd['MACDs_12_26_9']
        df['MACD_hist'] = macd['MACDh_12_26_9']
        
        # Other indicators
        df['ATR_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['TRIX_15'] = ta.trix(df['close'], length=15)['TRIX_15_9']
        df['PSAR'] = ta.psar(df['high'], df['low'], df['close'])['PSARl_0.02_0.2']
        
        # Volume features
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_pct'] = df['volume'].pct_change()
        
        # Fill any remaining NaNs with reasonable values
        df.fillna({
            'RSI_14': 50,
            'STOCH_%K': 50,
            'STOCH_%D': 50,
            'ADX_14': 25,
            'BB_%B': 0.5,
            'MACD_line': 0,
            'MACD_signal': 0,
            'MACD_hist': 0,
            'volatility': df['range'].median(),
            'volume_ma': df['volume'].median(),
            'volume_pct': 0
        }, inplace=True)
        
        # Calculate regimes based on SMAs
        conditions = [
            (df['close'] > df['SMA_50']) & (df['close'] > df['SMA_200']),  # Strong uptrend
            (df['close'] > df['SMA_50']) & (df['close'] <= df['SMA_200']), # Weak uptrend
            (df['close'] <= df['SMA_50']) & (df['close'] > df['SMA_200']), # Weak downtrend
            (df['close'] <= df['SMA_50']) & (df['close'] <= df['SMA_200']) # Strong downtrend
        ]
        
        for i in range(4):
            df[f'Regime{i}'] = conditions[i].astype(int)
    
    else:
        # Not enough data yet - fill with neutral values
        neutral_vals = {
            'RSI_14': 50,
            'STOCH_%K': 50,
            'STOCH_%D': 50,
            'ADX_14': 25,
            'BB_%B': 0.5,
            'MACD_line': 0,
            'MACD_signal': 0,
            'MACD_hist': 0,
            'volatility': df['range'].median(),
            'SMA_50': df['close'],
            'SMA_200': df['close'],
            'volume_ma': df['volume'].median(),
            'volume_pct': 0,
            'Regime0': 0, 'Regime1': 0, 'Regime2': 0, 'Regime3': 1,
            'ATR_14': df['range'].median(),
            'TRIX_15': 0,
            'PSAR': df['close'],
            'KC_upper': df['high'],
            'KC_middle': df['close'],
            'KC_lower': df['low'],
            'DMP_14': 0,
            'DMN_14': 0,
            'BB_width': 0
        }
        
        for col, val in neutral_vals.items():
            df[col] = val
    
    return df

def process_symbol(symbol):
    print(f"\nProcessing {symbol}...")
    
    try:
        # Read tick data
        tick_file = os.path.join(RAW_DATA_DIR, f"{symbol}_ticks.csv")
        tick_data = pd.read_csv(
            tick_file,
            usecols=['Time (EET)', 'Ask', 'Bid', 'AskVolume', 'BidVolume'],
            dtype={'Ask': float, 'Bid': float, 'AskVolume': float, 'BidVolume': float}
        )
    except Exception as e:
        print(f"Error reading {symbol} data: {str(e)}")
        return False

    # Convert EET to UTC
    try:
        eet = pytz.timezone('Europe/Bucharest')
        tick_data['timestamp'] = pd.to_datetime(tick_data['Time (EET)'], format='%Y.%m.%d %H:%M:%S.%f', errors='coerce')
        tick_data = tick_data.dropna(subset=['timestamp'])
        tick_data['timestamp'] = tick_data['timestamp'].dt.tz_localize(eet).dt.tz_convert(pytz.UTC)
        tick_data['timestamp'] = tick_data['timestamp'].dt.tz_localize(None)
    except Exception as e:
        print(f"Timestamp conversion failed: {str(e)}")
        return False

    # Filter by date range
    tick_data = tick_data[
        (tick_data['timestamp'] >= pd.to_datetime(START_DATE)) & 
        (tick_data['timestamp'] <= pd.to_datetime(END_DATE))
    ]
    
    if tick_data.empty:
        print(f"No data for {symbol} in date range {START_DATE} to {END_DATE}")
        return False

    # Calculate mid price and resample
    tick_data['mid'] = (tick_data['Ask'] + tick_data['Bid']) / 2
    tick_data['volume'] = tick_data['AskVolume'] + tick_data['BidVolume']
    
    try:
        ohlc = tick_data.resample('5min', on='timestamp').agg({
            'mid': ['first', 'max', 'min', 'last'],
            'volume': 'sum'
        })
        ohlc.columns = ['open', 'high', 'low', 'close', 'volume']
        ohlc = ohlc.dropna()
    except Exception as e:
        print(f"Resampling failed: {str(e)}")
        return False

    # Calculate all indicators
    ohlc = calculate_all_indicators(ohlc)

    # Time features
    ohlc['hour'] = ohlc.index.hour
    ohlc['day_of_week'] = ohlc.index.dayofweek

    # Ensure all required columns exist
    required_columns = [
        'open', 'high', 'low', 'close', 'volume',
        'RSI_14', 'BB_%B', 'ATR_14', 'STOCH_%K', 'STOCH_%D',
        'MACD_line', 'MACD_signal', 'KC_upper', 'KC_middle', 'KC_lower',
        'SMA_50', 'ADX_14', 'PSAR', 'SMA_200', 'TRIX_15',
        'Regime0', 'Regime1', 'Regime2', 'Regime3', 'volatility',
        'hour', 'day_of_week', 'momentum', 'volume_ma', 'volume_pct',
        'BB_width', 'MACD_hist', 'DMP_14', 'DMN_14'
    ]
    
    for col in required_columns:
        if col not in ohlc.columns:
            ohlc[col] = 0  # Fill missing with 0

    # Save processed data
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    output_file = os.path.join(PROCESSED_DIR, f"{symbol}_processed.csv")
    
    try:
        ohlc[required_columns].to_csv(output_file, index=True)
        print(f"Successfully processed {len(ohlc)} rows for {symbol}")
        return True
    except Exception as e:
        print(f"Failed to save {symbol}: {str(e)}")
        return False

def main():
    print(f"Processing data from {START_DATE} to {END_DATE}")
    print(f"Input directory: {RAW_DATA_DIR}")
    print(f"Output directory: {PROCESSED_DIR}")
    
    success_count = 0
    for symbol in SYMBOLS:
        if process_symbol(symbol):
            success_count += 1
    
    print("\nProcessing complete!")
    print(f"Successfully processed {success_count}/{len(SYMBOLS)} symbols")
    print(f"Processed files saved to: {PROCESSED_DIR}")

if __name__ == "__main__":
    main()