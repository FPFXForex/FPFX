# preprocess_data.py
# (all your original comments kept unchanged)

import os
import pandas as pd
import numpy as np
import pandas_ta as ta
from datetime import datetime
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import joblib  # Added to save HMM model
import pickle  # CRITICAL MISSING IMPORT ADDED

# 1) Configuration
SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "XAUUSD"]
INPUT_DIR = r"C:\FPFX\data"
OUTPUT_DIR = r"C:\FPFX\data\processed"
MODEL_DIR = r"C:\FPFX\model"  # Added to save HMM model
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)  # Ensure model directory exists

# 2) Timeframe definitions (in minutes)
TF_SETTINGS = {
    "M5": 5,
    "M15": 15,
    "H1": 60,
    "H4": 240,
    "D1": "D"
}

def read_ticks_to_mid(filepath):
    df = pd.read_csv(filepath, parse_dates=["time"])
    df["mid"] = (df["bid"] + df["ask"]) / 2.0
    df.set_index("time", inplace=True)
    df.index = pd.to_datetime(df.index, utc=True)
    return df[["mid"]]

def resample_ohlc(mid_df, timeframe):
    rule = "1D" if timeframe == "D" else f"{timeframe}min"
    ohlc = pd.DataFrame()
    ohlc["open"] = mid_df["mid"].resample(rule).first()
    ohlc["high"] = mid_df["mid"].resample(rule).max()
    ohlc["low"] = mid_df["mid"].resample(rule).min()
    ohlc["close"] = mid_df["mid"].resample(rule).last()
    ohlc["volume"] = mid_df["mid"].resample(rule).count()
    ohlc.dropna(subset=["open"], inplace=True)
    return ohlc

def compute_indicators(ohlc_df, tf_label):
    df = ohlc_df.copy()
    if tf_label == "M5":
        df["RSI_14"] = ta.rsi(df["close"], length=14)
        bb = ta.bbands(df["close"], length=20, std=2)
        df["BB_%B"] = bb["BBP_20_2.0"]
        df["ATR_14"] = ta.atr(df["high"], df["low"], df["close"], length=14)
        stoch = ta.stoch(df["high"], df["low"], df["close"], k=14, d=3, smooth_k=3)
        df["STOCH_%K"] = stoch["STOCHk_14_3_3"]
        df["STOCH_%D"] = stoch["STOCHd_14_3_3"]
    elif tf_label == "M15":
        macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
        df["MACD"] = macd["MACD_12_26_9"]
        df["MACD_signal"] = macd["MACDs_12_26_9"]
        kc = ta.kc(df["high"], df["low"], df["close"], length=20, scalar=2.0)
        kc_cols = kc.columns.tolist()
        df["KC_upper"] = kc[kc_cols[0]]
        df["KC_middle"] = kc[kc_cols[1]]
        df["KC_lower"] = kc[kc_cols[2]]
    elif tf_label == "H1":
        df["SMA_50"] = ta.sma(df["close"], length=50)
        df["ADX_14"] = ta.adx(df["high"], df["low"], df["close"], length=14)["ADX_14"]
        df["PSAR"] = ta.psar(df["high"], df["low"], df["close"], step=0.02, max_step=0.2)["PSARl_0.02_0.2"]
    elif tf_label == "H4":
        ichi = ta.ichimoku(df["high"], df["low"], df["close"], tenkan=9, kijun=26, senkou=52)
        if isinstance(ichi, pd.DataFrame):
            df["Ichimoku_Tenkan"] = ichi["ITS_9_26_52"]
            df["Ichimoku_Kijun"] = ichi["IKS_9_26_52"]
            df["Ichimoku_Senkou_A"] = ichi["ISA_9_26_52"]
            df["Ichimoku_Senkou_B"] = ichi["ISB_9_26_52"]
        trix_df = ta.trix(df["close"], length=15)
        if isinstance(trix_df, pd.DataFrame):
            trix_col = trix_df.columns[0]
            df["TRIX_15"] = trix_df[trix_col]
        else:
            df["TRIX_15"] = trix_df
    elif tf_label == "D1":
        df["SMA_200"] = ta.sma(df["close"], length=200)
        df["VWMA_20"] = ta.vwma(df["close"], df["volume"], length=20)
    return df

def compute_hmm_regimes(h1_df):
    df = h1_df.copy().dropna(subset=["ATR_14", "ADX_14", "SMA_50", "SMA_200"])
    df["SMA_spread"] = df["SMA_50"] - df["SMA_200"]
    features = df[["ATR_14", "ADX_14", "SMA_spread"]].values

    scaler = StandardScaler()
    scaled_feats = scaler.fit_transform(features)
    mask = ~np.isnan(scaled_feats).any(axis=1)
    scaled_feats = scaled_feats[mask]
    valid_index = df.index[mask]

    model = hmm.GaussianHMM(n_components=4, covariance_type="full", n_iter=1000, random_state=42)
    model.fit(scaled_feats)
    probs = model.predict_proba(scaled_feats)
    
    # SAVE HMM MODEL FOR LIVE TRADING (CRITICAL ADDITION)
    hmm_model_path = os.path.join(MODEL_DIR, "H1_hmm_model.pkl")
    with open(hmm_model_path, "wb") as f:
        pickle.dump({"hmm_model": model, "scaler": scaler}, f)
    print(f"Saved HMM model to {hmm_model_path}")

    regimes_df = pd.DataFrame(data=probs, index=valid_index, columns=[f"Regime{i}" for i in range(4)])
    return regimes_df

def merge_timeframes(symbol):
    ticks_path = os.path.join(INPUT_DIR, f"{symbol}_ticks.csv")
    print(f"Processing {symbol} ticks from:\n  {ticks_path}")
    mid_df = read_ticks_to_mid(ticks_path)

    ohlc = {}
    for tf_label, tf_value in TF_SETTINGS.items():
        ohlc[tf_label] = resample_ohlc(mid_df, tf_value)
        print(f"  {symbol}: {tf_label} bars = {len(ohlc[tf_label])} rows")

    ind = {}
    for tf_label in TF_SETTINGS.keys():
        ind[tf_label] = compute_indicators(ohlc[tf_label], tf_label)
        print(f"  {symbol}: {tf_label} indicators computed")

    # Forward-fill SMA_200 from D1 into H1
    d1 = ind["D1"][["SMA_200"]].dropna()
    sma200_on_h1 = d1["SMA_200"].reindex(ind["H1"].index, method="ffill")
    ind["H1"]["SMA_200"] = sma200_on_h1

    # Forward-fill ATR_14 from M5 into H1
    atr14_on_h1 = ind["M5"]["ATR_14"].reindex(ind["H1"].index, method="ffill")
    ind["H1"]["ATR_14"] = atr14_on_h1

    # Compute HMM regimes
    regimes_h1 = compute_hmm_regimes(ind["H1"][["ATR_14", "ADX_14", "SMA_50", "SMA_200"]])
    print(f"  {symbol}: HMM regimes computed, shape = {regimes_h1.shape}")

    # Merge all indicators onto M5 base
    df_final = ind["M5"].copy()
    for col in ind["M15"].columns:
        df_final[col] = ind["M15"][col].reindex(df_final.index, method="ffill")
    for col in ind["H1"].columns:
        df_final[col] = ind["H1"][col].reindex(df_final.index, method="ffill")
    for col in ind["H4"].columns:
        df_final[col] = ind["H4"][col].reindex(df_final.index, method="ffill")
    for col in ind["D1"].columns:
        df_final[col] = ind["D1"][col].reindex(df_final.index, method="ffill")
    for col in regimes_h1.columns:
        df_final[col] = regimes_h1[col].reindex(df_final.index, method="ffill")

    df_final.dropna(subset=["RSI_14", "BB_%B", "ATR_14", "STOCH_%K", "STOCH_%D"], inplace=True)
    out_path = os.path.join(OUTPUT_DIR, f"{symbol}_processed.csv")
    df_final.to_csv(out_path, index_label="time")
    print(f"  {symbol}: final DataFrame shape = {df_final.shape}, saved to:\n    {out_path}\n")

if __name__ == "__main__":
    for sym in SYMBOLS:
        merge_timeframes(sym)
    print("All symbols processed and saved.")