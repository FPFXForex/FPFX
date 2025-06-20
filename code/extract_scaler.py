import os
import joblib
import pandas as pd
import time
from train import ForexMultiEnv, ALL_DATA, SYMBOLS  # Import from your existing training script

# ========== CONFIGURATION ==========
MODEL_DIR = "C:/FPFX/model"
DATA_DIR = "C:/FPFX/Data/processed"
NEWS_CSV = "C:/FPFX/Data/news_cache.csv"
os.makedirs(MODEL_DIR, exist_ok=True)

def extract_scaler():
    print("\n=== Forex AI Scaler Extraction ===")
    print(f"Model Directory: {MODEL_DIR}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Symbols: {', '.join(SYMBOLS)}")
    
    start_time = time.time()
    
    print("\n[1/3] Loading environment and data...")
    env = ForexMultiEnv(ALL_DATA)  # This uses your existing data loading logic
    
    print("\n[2/3] Scaler Summary:")
    print(f"Features scaled: {len(env.scaler.mean_)}")
    print(f"Example means: {env.scaler.mean_[:3]}...")
    print(f"Example scales: {env.scaler.scale_[:3]}...")
    
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    joblib.dump(env.scaler, scaler_path)
    
    elapsed = time.time() - start_time
    print(f"\n[3/3] Scaler saved to {scaler_path}")
    print(f"Completed in {elapsed:.2f} seconds")

if __name__ == "__main__":
    extract_scaler()
    print("\nOperation complete. Safe to close this window.")