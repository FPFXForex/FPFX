import MetaTrader5 as mt5
# Attempt to initialize a connection to the running MT5 terminal
if not mt5.initialize():
    print("MT5 initialize() failed, error code =", mt5.last_error())
else:
    print("MT5 initialized successfully")
    mt5.shutdown()

