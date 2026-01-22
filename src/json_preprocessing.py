import pandas as pd
import json
import ta
from ta.utils import dropna

from json_manipulation import get_json_results
    
"""
Clean df by dropping unnecessary columns
"""
def clean_df(df: pd.DataFrame, cols_to_drop: list) -> pd.DataFrame:
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    
    return df


"""
Adds timestamps to df (UTC to EST); Filters to RTH if specified

@param df: pd.DataFrame - input DataFrame
@param rth bool: filter to regular trading hours only (True by default)
@returns: pd.DataFrame
"""
def add_timestamps(df: pd.DataFrame, rth=True) -> pd.DataFrame:
    
    df["datetime"] = pd.to_datetime(df["UTC_timestamp"], unit="ms", utc=True)
    df["datetime_est"] = df["datetime"].dt.tz_convert("America/New_York")
    
    df = df.set_index("datetime_est")
    
    if rth:
        df = df.between_time("09:30", "16:00")
    
    return df


"""
Adds momentum baseed indicators to df

@param df: pd.DataFrame = input DataFrame
@returns: pd.DataFrame
"""
def add_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:

    # Rolling Windows
        # SMAs (Standard)

    df['sma_5_rolling'] = ta.trend.sma_indicator(df['close'], window=5)
    df["sma_10_rolling"] = ta.trend.sma_indicator(df['close'], window=10)
    df['sma_20_rolling'] = ta.trend.sma_indicator(df['close'], window=20)
    df["sma_25_rolling"] = ta.trend.sma_indicator(df['close'], window=25)
    df['sma_50_rolling'] = ta.trend.sma_indicator(df['close'], window=50)
    
        #SMAs (Custom)
    df["sma_3_rolling"] = ta.trend.sma_indicator(df['close'], window=3) #15 min
    df["sma_6_rolling"] = ta.trend.sma_indicator(df['close'], window=6) #30 min
    df["sma_9_rolling"] = ta.trend.sma_indicator(df['close'], window=9) #45 min
    df["sma_12_rolling"] = ta.trend.sma_indicator(df['close'], window=12) #60 min
    
        #EMAs (Standard)
    df['ema_5_rolling'] = ta.trend.ema_indicator(df['close'], window=5)
    df['ema_10_rolling'] = ta.trend.ema_indicator(df['close'], window=10)
    df['ema_20_rolling'] = ta.trend.ema_indicator(df['close'], window=20)
    df['ema_25_rolling'] = ta.trend.ema_indicator(df['close'], window=25)
    df['ema_50_rolling'] = ta.trend.ema_indicator(df['close'], window=50)

        #EMAs (Custom)
    df["ema_3_rolling"] = ta.trend.ema_indicator(df['close'], window=3) #15 min
    df["ema_6_rolling"] = ta.trend.ema_indicator(df['close'], window=6) #30 min
    df["ema_9_rolling"] = ta.trend.ema_indicator(df['close'], window=9) #45 min
    df["ema_12_rolling"] = ta.trend.ema_indicator(df['close'], window=12) #60 min
    
        #RSI (Standard)
    df['rsi_7_rolling'] = ta.momentum.rsi(df['close'], window=7)
    df['rsi_14_rolling'] = ta.momentum.rsi(df['close'], window=14)
    df['rsi_21_rolling'] = ta.momentum.rsi(df['close'], window=21)
    
        #RSI (Custom)
    df['rsi_3_rolling'] = ta.momentum.rsi(df['close'], window=3) #15 min
    df['rsi_6_rolling'] = ta.momentum.rsi(df['close'], window=6) #30 min
    df['rsi_9_rolling'] = ta.momentum.rsi(df['close'], window=9) #45 min
    df['rsi_12_rolling'] = ta.momentum.rsi(df['close'], window=12) #60 min
    
    
    # Isolated Windows    
    
    assert df.index.tz is not None
    df["trade_date"] = df.index.date
    

        #SMAs (Standard)
    df['sma_5_isolated'] = df.groupby("trade_date", group_keys=False)["close"].apply(lambda x: ta.trend.sma_indicator(x, window=5))
    df["sma_10_isolated"] = df.groupby("trade_date", group_keys=False)["close"].apply(lambda x: ta.trend.sma_indicator(x, window=10))
    df['sma_20_isolated'] = df.groupby("trade_date", group_keys=False)["close"].apply(lambda x: ta.trend.sma_indicator(x, window=20))
    df["sma_25_isolated"] = df.groupby("trade_date", group_keys=False)["close"].apply(lambda x: ta.trend.sma_indicator(x, window=25))
    df['sma_50_isolated'] = df.groupby("trade_date", group_keys=False)["close"].apply(lambda x: ta.trend.sma_indicator(x, window=50))
    
        #SMAs (Custom)
    df['sma_3_isolated'] = df.groupby("trade_date", group_keys=False)["close"].apply(lambda x: ta.trend.sma_indicator(x, window=3)) #15 min
    df["sma_6_isolated"] = df.groupby("trade_date", group_keys=False)["close"].apply(lambda x: ta.trend.sma_indicator(x, window=6)) #30 min
    df['sma_9_isolated'] = df.groupby("trade_date", group_keys=False)["close"].apply(lambda x: ta.trend.sma_indicator(x, window=9)) #45 min
    df['sma_12_isolated'] = df.groupby("trade_date", group_keys=False)["close"].apply(lambda x: ta.trend.sma_indicator(x, window=12)) #60 min


        #EMAs (Standard)
    df['ema_5_isolated'] = df.groupby("trade_date", group_keys=False)["close"].apply(lambda x: ta.trend.ema_indicator(x, window=5))
    df['ema_10_isolated'] = df.groupby("trade_date", group_keys=False)["close"].apply(lambda x: ta.trend.ema_indicator(x, window=10))
    df['ema_20_isolated'] = df.groupby("trade_date", group_keys=False)["close"].apply(lambda x: ta.trend.ema_indicator(x, window=20))
    df['ema_25_isolated'] = df.groupby("trade_date", group_keys=False)["close"].apply(lambda x: ta.trend.ema_indicator(x, window=25))   
    df['ema_50_isolated'] = df.groupby("trade_date", group_keys=False)["close"].apply(lambda x: ta.trend.ema_indicator(x, window=50)) 
    
        #EMAs (Custom)
    df['ema_3_isolated'] = df.groupby("trade_date", group_keys=False)["close"].apply(lambda x: ta.trend.ema_indicator(x, window=3)) #15 min
    df['ema_6_isolated'] = df.groupby("trade_date", group_keys=False)["close"].apply(lambda x: ta.trend.ema_indicator(x, window=6)) #30 min
    df['ema_9_isolated'] = df.groupby("trade_date", group_keys=False)["close"].apply(lambda x: ta.trend.ema_indicator(x, window=9)) #45 min
    df['ema_12_isolated'] = df.groupby("trade_date", group_keys=False)["close"].apply(lambda x: ta.trend.ema_indicator(x, window=12)) #60 min

        #RSI (Standard)
    df['rsi_7_isolated'] = df.groupby("trade_date", group_keys=False)["close"].apply(lambda x: ta.momentum.rsi(x, window=7))
    df['rsi_14_isolated'] = df.groupby("trade_date", group_keys=False)["close"].apply(lambda x: ta.momentum.rsi(x, window=14))
    df['rsi_21_isolated'] = df.groupby("trade_date", group_keys=False)["close"].apply(lambda x: ta.momentum.rsi(x, window=21))
    
        #RSI (Custom)
    df['rsi_3_isolated'] = df.groupby("trade_date", group_keys=False)["close"].apply(lambda x: ta.momentum.rsi(x, window=3)) #15 min
    df['rsi_6_isolated'] = df.groupby("trade_date", group_keys=False)["close"].apply(lambda x: ta.momentum.rsi(x, window=6)) #30 min
    df['rsi_9_isolated'] = df.groupby("trade_date", group_keys=False)["close"].apply(lambda x: ta.momentum.rsi(x, window=9)) #45 min
    df['rsi_12_isolated'] = df.groupby("trade_date", group_keys=False)["close"].apply(lambda x: ta.momentum.rsi(x, window=12)) #60 min

    df.drop(columns=["trade_date"], inplace=True)

    return df



def add_volume_indicator(df: pd.DataFrame) -> pd.DataFrame:
    
    # Rolling Windows
        # VWAP (Standard)
        
    df["vwap_7_rolling"] = ta.volume.volume_weighted_average_price(df['high'], df['low'], df['close'], df['volume'], window=7)
    df["vwap_14_rolling"] = ta.volume.volume_weighted_average_price(df['high'], df['low'], df['close'], df['volume'], window=14)
    df["vwap_21_rolling"] = ta.volume.volume_weighted_average_price(df['high'], df['low'], df['close'], df['volume'], window=21)

        #VWAP (Custom)
    df["vwap_3_rolling"] = ta.volume.volume_weighted_average_price(df['high'], df['low'], df['close'], df['volume'], window=3) #15 min
    df["vwap_6_rolling"] = ta.volume.volume_weighted_average_price(df['high'], df['low'], df['close'], df['volume'], window=6) #30 min
    df["vwap_9_rolling"] = ta.volume.volume_weighted_average_price(df['high'], df['low'], df['close'], df['volume'], window=9) #45 min
    df["vwap_12_rolling"] = ta.volume.volume_weighted_average_price(df['high'], df['low'], df['close'], df['volume'], window=12) #60 min
    
    
    return df




def add_ta_indicators(df: pd.DataFrame) -> pd.DataFrame:

    df = add_momentum_indicators(df)
    
    return df



"""
Preprocess data
"""
def preprocess_data(df: pd.DataFrame, cols_to_drop: list = [], rth:bool =True) -> pd.DataFrame:
    

    
    df = add_timestamps(df, rth)
    df = add_ta_indicators(df)
    
    df = clean_df(df, cols_to_drop=cols_to_drop)
    
    return df


"""
Testing Area
"""
if __name__ == "__main__":
    
    df = get_json_results("assets/raw/2025_5_minute/annual_2025_5_minute_SPY.json")
    
    cols_to_drop = ["volume_weighted_average_price", "UTC_timestamp", "datetime"]

    df = preprocess_data(df, cols_to_drop=cols_to_drop, rth=True)

    
    print(df)
    print(df.columns)
    
    