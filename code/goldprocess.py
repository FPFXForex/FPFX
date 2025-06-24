import os
import pandas as pd
import pandas_ta as ta
import pytz
from datetime import datetime
import numpy as np

# Configuration
RAW_DATA_DIR = "C:/FPFX/Data"
PROCESSED_DIR = "C:/FPFX/Data/Processed"
SYMBOL = "XAUUSD"
START_DATE = "2023-06-12"
END_DATE = "2025-06-12"
CHUNK_SIZE = 1_000_000  # Process 1 million rows at a time

def calculate_all_indicators(df):
    """Calculate all technical indicators with proper warm-up period"""
    df = df.copy()
    
    # Remove duplicate indices if any exist
    df = df[~df.index.duplicated(keep='first')]
    
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
        
        # Oscillators - handle potential NaN values
        try:
            df['RSI_14'] = ta.rsi(df['close'], length=14).fillna(50)
            stoch = ta.stoch(df['high'], df['low'], df['close'])
            df['STOCH_%K'] = stoch['STOCHk_14_3_3'].fillna(50)
            df['STOCH_%D'] = stoch['STOCHd_14_3_3'].fillna(50)
        except:
            df['RSI_14'] = 50
            df['STOCH_%K'] = 50
            df['STOCH_%D'] = 50
        
        # Trend indicators
        try:
            adx_data = ta.adx(df['high'], df['low'], df['close'], length=14)
            df['ADX_14'] = adx_data['ADX_14'].fillna(25)
            df['DMP_14'] = adx_data['DMP_14'].fillna(0)
            df['DMN_14'] = adx_data['DMN_14'].fillna(0)
        except:
            df['ADX_14'] = 25
            df['DMP_14'] = 0
            df['DMN_14'] = 0
        
        # Bands
        try:
            bb = ta.bbands(df['close'], length=20)
            df['BB_%B'] = (df['close'] - bb['BBL_20_2.0']) / (bb['BBU_20_2.0'] - bb['BBL_20_2.0']).replace(0, 1e-10).fillna(0.5)
            df['BB_width'] = (bb['BBU_20_2.0'] - bb['BBL_20_2.0']) / bb['BBM_20_2.0'].fillna(0)
        except:
            df['BB_%B'] = 0.5
            df['BB_width'] = 0
        
        try:
            kc = ta.kc(df['high'], df['low'], df['close'])
            df['KC_upper'] = kc['KCUe_20_2'].fillna(df['high'])
            df['KC_middle'] = kc['KCBe_20_2'].fillna(df['close'])
            df['KC_lower'] = kc['KCLe_20_2'].fillna(df['low'])
        except:
            df['KC_upper'] = df['high']
            df['KC_middle'] = df['close']
            df['KC_lower'] = df['low']
        
        # MACD
        try:
            macd = ta.macd(df['close'])
            df['MACD_line'] = macd['MACD_12_26_9'].fillna(0)
            df['MACD_signal'] = macd['MACDs_12_26_9'].fillna(0)
            df['MACD_hist'] = macd['MACDh_12_26_9'].fillna(0)
        except:
            df['MACD_line'] = 0
            df['MACD_signal'] = 0
            df['MACD_hist'] = 0
        
        # Other indicators
        try:
            df['ATR_14'] = ta.atr(df['high'], df['low'], df['close'], length=14).fillna(df['range'].median())
            df['TRIX_15'] = ta.trix(df['close'], length=15)['TRIX_15_9'].fillna(0)
            df['PSAR'] = ta.psar(df['high'], df['low'], df['close'])['PSARl_0.02_0.2'].fillna(df['close'])
        except:
            df['ATR_14'] = df['range'].median()
            df['TRIX_15'] = 0
            df['PSAR'] = df['close']
        
        # Volume features
        df['volume_ma'] = df['volume'].rolling(20).mean().fillna(df['volume'].median())
        df['volume_pct'] = df['volume'].pct_change().fillna(0)
        
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

def process_xauusd():
    print(f"\nProcessing {SYMBOL}...")
    
    try:
        # Process data in chunks
        tick_file = os.path.join(RAW_DATA_DIR, f"{SYMBOL}_ticks.csv")
        
        # Initialize empty DataFrame for results
        ohlc_list = []
        
        # Process data in chunks
        for chunk in pd.read_csv(
            tick_file,
            usecols=['Time (EET)', 'Ask', 'Bid', 'AskVolume', 'BidVolume'],
            dtype={'Ask': 'float32', 'Bid': 'float32', 'AskVolume': 'float32', 'BidVolume': 'float32'},
            chunksize=CHUNK_SIZE
        ):
            # Convert and filter dates
            chunk['Time (EET)'] = pd.to_datetime(chunk['Time (EET)'], format='%Y.%m.%d %H:%M:%S.%f', errors='coerce')
            chunk = chunk.dropna(subset=['Time (EET)'])
            
            # Filter by date range
            mask = (chunk['Time (EET)'] >= pd.to_datetime(START_DATE)) & (chunk['Time (EET)'] <= pd.to_datetime(END_DATE))
            chunk = chunk[mask]
            
            if chunk.empty:
                continue
                
            # Convert timezone
            chunk['timestamp'] = chunk['Time (EET)'].dt.tz_localize('Europe/Bucharest').dt.tz_convert('UTC').dt.tz_localize(None)
            
            # Calculate mid price and volume
            chunk['mid'] = (chunk['Ask'] + chunk['Bid']) / 2
            chunk['volume'] = chunk['AskVolume'] + chunk['BidVolume']
            
            # Resample to 5min
            chunk_ohlc = chunk.resample('5min', on='timestamp').agg({
                'mid': ['first', 'max', 'min', 'last'],
                'volume': 'sum'
            })
            chunk_ohlc.columns = ['open', 'high', 'low', 'close', 'volume']
            chunk_ohlc = chunk_ohlc.dropna()
            
            if not chunk_ohlc.empty:
                ohlc_list.append(chunk_ohlc)
                
    except Exception as e:
        print(f"Error processing data: {str(e)}")
        return False

    if not ohlc_list:
        print("No data processed")
        return False
        
    # Combine all processed chunks and remove any duplicate indices
    ohlc = pd.concat(ohlc_list)
    ohlc = ohlc[~ohlc.index.duplicated(keep='first')]
    ohlc = ohlc.sort_index()
    
    # Calculate indicators on the full dataset
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
            ohlc[col] = 0

    # Save processed data
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    output_file = os.path.join(PROCESSED_DIR, f"{SYMBOL}_processed.csv")
    
    try:
        # Save in chunks if needed
        ohlc.to_csv(output_file, index=True)
        print(f"Successfully processed {len(ohlc)} rows for {SYMBOL}")
        return True
    except Exception as e:
        print(f"Failed to save {SYMBOL}: {str(e)}")
        return False

if __name__ == "__main__":
    print(f"Processing {SYMBOL} data from {START_DATE} to {END_DATE}")
    print(f"Input directory: {RAW_DATA_DIR}")
    print(f"Output directory: {PROCESSED_DIR}")
    
    if process_xauusd():
        print("\nProcessing complete!")
        print(f"Processed file saved to: {os.path.join(PROCESSED_DIR, f'{SYMBOL}_processed.csv')}")
    else:
        print("\nProcessing failed")