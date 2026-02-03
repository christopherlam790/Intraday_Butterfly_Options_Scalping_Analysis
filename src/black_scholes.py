
import pandas as pd

if __name__ == "__main__":
    

    df = pd.read_csv(
        "assets/raw/options_data/opra-pillar-20250131-20260130.ohlcv-1m.csv",
        usecols=["ts_event", "symbol", "open", "high", "low", "close", "volume"],
        parse_dates=["ts_event"]
    )
    
    print(df)
    pass