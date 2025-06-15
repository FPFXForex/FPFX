# download_ticks.py
#
# This script connects to a running MetaTrader 5 terminal (already logged into ICMarkets-Demo)
# and downloads 5 years of tick data, day by day, for six symbols:
# EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, XAUUSD.
# Each symbol’s ticks are written to a separate CSV file in C:\FPFX\data\.
#
# IMPORTANT: You must:
# 1. Have MT5 running and logged in with your IC Markets demo credentials.
# 2. Run this script from within the (venv) environment so that `MetaTrader5` is importable.
# 3. Be patient—downloading 5 years of tick data for 6 instruments can take several hours.

import MetaTrader5 as mt5
import pandas as pd
import os
from datetime import datetime, timedelta, timezone

# 1) MT5 initialization parameters
MT5_LOGIN = 52380678      # e.g. 12345678
MT5_PASSWORD = "OlN&E$83kbSNYX"     # e.g. "Abc12345!"
MT5_SERVER = "ICMarketsSC-Demo"    # e.g. "ICMarkets-Demo03"

# 2) Symbols to download
SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "XAUUSD"]

# 3) Time range: last 5 years from today
END_UTC = datetime.now(timezone.utc)
START_UTC = END_UTC - timedelta(days=5*365)  # approx 5 years ago

# 4) Directory to store CSVs
OUTPUT_DIR = r"C:\FPFX\data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 5) Initialize MT5
if not mt5.initialize(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
    print("---- ERROR: MT5 initialization failed ----")
    print("Last error:", mt5.last_error())
    mt5.shutdown()
    exit(1)
print("MT5 initialized successfully")

# 6) For each symbol, loop day by day
for symbol in SYMBOLS:
    print(f"\nDownloading ticks for {symbol} ...")
    # Prepare output CSV path
    csv_path = os.path.join(OUTPUT_DIR, f"{symbol}_ticks.csv")
    # If file does not exist, write header now
    if not os.path.isfile(csv_path):
        header_df = pd.DataFrame(columns=["time", "bid", "ask"])
        header_df.to_csv(csv_path, index=False)

    # Start day-by-day loop
    current_start = START_UTC.replace(hour=0, minute=0, second=0, microsecond=0)
    one_day = timedelta(days=1)

    while current_start < END_UTC:
        current_end = current_start + one_day

        # Fetch ticks for this 24-hour window
        ticks = mt5.copy_ticks_range(symbol, current_start, current_end, mt5.COPY_TICKS_ALL)
        if ticks is None:
            print(f"  {symbol}: no data returned on {current_start.date()}, skipping.")
        else:
            # Convert to DataFrame
            df = pd.DataFrame(ticks)
            if not df.empty:
                # Convert 'time' (POSIX) to UTC datetime
                df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
                # Keep only 'time', 'bid', 'ask'
                df = df[["time", "bid", "ask"]]
                # Append (without header)
                df.to_csv(csv_path, mode="a", header=False, index=False)
                print(f"  {symbol}: wrote {len(df)} ticks for {current_start.date()}")

        current_start = current_end  # move to next day

    print(f"Finished downloading {symbol}. CSV saved to:\n  {csv_path}")

# 7) Shutdown MT5
mt5.shutdown()
print("\nAll symbols downloaded. MT5 shut down. Script complete.")
