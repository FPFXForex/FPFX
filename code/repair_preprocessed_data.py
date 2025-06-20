import pandas as pd
import pandas_ta as ta
import os
from tqdm import tqdm

DATA_DIR = "C:/FPFX/Data/processed"
SYMBOLS = ["AUDUSD", "EURUSD", "GBPUSD", "USDCAD", "USDJPY", "XAUUSD"]

def safe_calculate(df, calc_func, col_name, *args, **kwargs):
    """Safely calculate an indicator and align with original data"""
    try:
        # Calculate indicator
        result = calc_func(*args, **kwargs)
        
        if result is not None:
            # Handle different return types
            if isinstance(result, pd.DataFrame):
                # For multi-column outputs (like MACD, ADX)
                for col in result.columns:
                    new_col = f"{col_name}_{col}" if col_name not in col else col
                    df[new_col] = result[col]
            elif isinstance(result, pd.Series):
                # For single column outputs
                df[col_name] = result
            else:
                # For scalar or other outputs
                df[col_name] = result
                
            # Forward fill any NaN values that might occur at beginning of series
            for col in [c for c in df.columns if col_name in c]:
                df[col] = df[col].ffill()
                
        return df
    except Exception as e:
        print(f"⚠️ Error calculating {col_name}: {str(e)}")
        return df

def calculate_indicators(df):
    """Calculate all indicators with proper alignment"""
    df = df.copy()
    original_length = len(df)
    
    # Ensure datetime index for proper calculation
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        df = df.set_index('time')
    
    # MACD (returns MACD_0, MACD_1, MACD_2)
    df = safe_calculate(df, ta.macd, "MACD", 
                       df['close'], fast=12, slow=26, signal=9)
    if all(x in df.columns for x in ['MACD_0', 'MACD_1']):
        df['MACD_line'] = df['MACD_0']
        df['MACD_signal'] = df['MACD_1']
        df.drop(['MACD_0', 'MACD_1', 'MACD_2'], axis=1, inplace=True, errors='ignore')
    
    # Keltner Channels (returns KC_0, KC_1, KC_2)
    df = safe_calculate(df, ta.kc, "KC", 
                       df['high'], df['low'], df['close'], length=20, scalar=2)
    if all(x in df.columns for x in ['KC_0', 'KC_1', 'KC_2']):
        df['KC_upper'] = df['KC_0']
        df['KC_middle'] = df['KC_1']
        df['KC_lower'] = df['KC_2']
        df.drop(['KC_0', 'KC_1', 'KC_2'], axis=1, inplace=True, errors='ignore')
    
    # SMAs
    df = safe_calculate(df, ta.sma, "SMA_50", df['close'], length=50)
    df = safe_calculate(df, ta.sma, "SMA_200", df['close'], length=200)
    
    # ADX (returns ADX_0, ADX_1, ADX_2)
    df = safe_calculate(df, ta.adx, "ADX", 
                       df['high'], df['low'], df['close'], length=14)
    if 'ADX_0' in df.columns:
        df['ADX_14'] = df['ADX_0']
        df.drop(['ADX_0', 'ADX_1', 'ADX_2'], axis=1, inplace=True, errors='ignore')
    
    # PSAR
    df = safe_calculate(df, ta.psar, "PSAR", 
                       df['high'], df['low'], af0=0.02, af=0.02, max_af=0.2)
    
    # TRIX
    df = safe_calculate(df, ta.trix, "TRIX_15", df['close'], length=15)
    
    # Reset index if we set it earlier
    if 'time' not in df.columns and df.index.name == 'time':
        df = df.reset_index()
    
    # Ensure all columns match original length and fill any remaining NaNs
    for col in df.columns:
        if df[col].isna().any():
            df[col] = df[col].ffill().bfill()
    
    return df

def repair_files():
    """Process all files with guaranteed proper output"""
    for symbol in tqdm(SYMBOLS, desc="Processing"):
        file_path = os.path.join(DATA_DIR, f"{symbol}_processed.csv")
        if not os.path.exists(file_path):
            print(f"\n⚠️ Missing: {file_path}")
            continue
        
        try:
            # Read with low_memory=False to prevent mixed type warnings
            df = pd.read_csv(file_path, low_memory=False)
            
            # Calculate indicators
            df_fixed = calculate_indicators(df)
            
            # Target columns in desired order
            target_cols = [
                'time','open','high','low','close','volume',
                'RSI_14','BB_%B','ATR_14','STOCH_%K','STOCH_%D',
                'MACD_line','MACD_signal',
                'KC_upper','KC_middle','KC_lower',
                'SMA_50','ADX_14','PSAR','SMA_200','TRIX_15',
                'Regime0','Regime1','Regime2','Regime3','volatility'
            ]
            
            # Save only existing columns in correct order
            final_cols = [c for c in target_cols if c in df_fixed.columns]
            
            # Ensure we don't lose any existing columns
            existing_cols = [c for c in df.columns if c not in target_cols]
            final_cols.extend(existing_cols)
            
            # Save backup before overwriting
            backup_path = file_path.replace(".csv", "_backup.csv")
            df.to_csv(backup_path, index=False)
            
            # Save the fixed file
            df_fixed[final_cols].to_csv(file_path, index=False)
            
            # Report changes
            new_cols = set(df_fixed.columns) - set(df.columns)
            if new_cols:
                print(f"\n✅ {symbol}: Added {len(new_cols)} indicators: {', '.join(new_cols)}")
            else:
                print(f"\nℹ️ {symbol}: No new indicators added (all were already present)")
                
        except Exception as e:
            print(f"\n❌ {symbol} failed: {str(e)}")

if __name__ == "__main__":
    print("=== INDICATOR REPAIR TOOL ===")
    print(f"Processing directory: {DATA_DIR}\n")
    repair_files()
    print("\n=== COMPLETED ===")