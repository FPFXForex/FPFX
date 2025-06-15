import os
import pandas as pd
from glob import glob

# Folder containing downloaded raw CSVs
input_folder = r"C:\FPFX\data\raw"
output_folder = r"C:\FPFX\data"

# List of expected symbols
symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "XAUUSD"]

for symbol in symbols:
    pattern = os.path.join(input_folder, f"{symbol}_Ticks_*.csv")
    files = glob(pattern)
    if not files:
        print(f"❌ No file found for {symbol}")
        continue

    filepath = files[0]
    print(f"🔄 Processing {os.path.basename(filepath)}")

    df = pd.read_csv(
        filepath,
        usecols=["Time (EET)", "Ask", "Bid"],  # we discard volumes
        parse_dates=["Time (EET)"],
        dayfirst=False,
    )

    # Rename and reorder columns
    df = df.rename(columns={
        "Time (EET)": "time",
        "Bid": "bid",
        "Ask": "ask"
    })[["time", "bid", "ask"]]

    # Write cleaned CSV for pipeline
    out_path = os.path.join(output_folder, f"{symbol}_ticks.csv")
    df.to_csv(out_path, index=False)
    print(f"✅ Saved cleaned: {out_path} ({len(df):,} rows)")