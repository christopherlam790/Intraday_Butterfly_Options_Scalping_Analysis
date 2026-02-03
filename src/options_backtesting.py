
import pandas as pd



if __name__ == "__main__":
        
    # Load OHLCV
    ohlcv = pd.read_csv(
        "assets/raw/options_data/opra-pillar-20250131-20260130.ohlcv-1m.csv",
        parse_dates=["ts_event"]
    )

    # Remove spaces from symbol for easier slicing
    ohlcv["symbol_clean"] = ohlcv["symbol"].str.replace(" ", "", regex=False)

    # Extract expiration date (YYMMDD -> datetime)
    ohlcv["expiration"] = pd.to_datetime(
        ohlcv["symbol_clean"].str[3:9], format="%y%m%d"
    )

    # Make expiration timezone-aware, UTC
    ohlcv["expiration"] = ohlcv["expiration"].dt.tz_localize("UTC")


    # Call/Put flag
    ohlcv["cp_flag"] = ohlcv["symbol_clean"].str[9]

    # Strike price (last 8 digits / 1000)
    ohlcv["strike"] = ohlcv["symbol_clean"].str[10:].astype(int) / 1000.0

    # Underlying (first 3 chars)
    ohlcv["underlying"] = ohlcv["symbol_clean"].str[:3]

    ohlcv["dte"] = (ohlcv["expiration"] - ohlcv["ts_event"]).dt.days

    df_backtest = ohlcv[
        (ohlcv["underlying"].isin(["SPX", "SPXW"])) &
        (ohlcv["dte"] == 0)
    ].copy()


    print(df_backtest)


    pass