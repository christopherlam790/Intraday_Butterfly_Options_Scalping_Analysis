import pandas as pd
import ta
from ta.utils import dropna
import math
import numpy as np

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
Adds momentum based indicators (RSI, ROC) to df

@param df: pd.DataFrame = input DataFrame
@returns: pd.DataFrame
"""
def add_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:

    # Isolated Windows    
    
    assert df.index.tz is not None
    df["trade_date"] = df.index.date
           
        #RSI (Custom)
    df['rsi_3_isolated'] = df.groupby("trade_date", group_keys=False)["close"].apply(lambda x: ta.momentum.rsi(x, window=3)) #15 min
    df['rsi_6_isolated'] = df.groupby("trade_date", group_keys=False)["close"].apply(lambda x: ta.momentum.rsi(x, window=6)) #30 min
    df['rsi_12_isolated'] = df.groupby("trade_date", group_keys=False)["close"].apply(lambda x: ta.momentum.rsi(x, window=12)) #60 min

        #ROC (Custom)
    df['roc_3_isolated'] = df.groupby("trade_date", group_keys=False)["close"].apply(lambda x: ta.momentum.roc(x, window=3)) #15 min
    df['roc_6_isolated'] = df.groupby("trade_date", group_keys=False)["close"].apply(lambda x: ta.momentum.roc(x, window=6)) #30 min
    df['roc_12_isolated'] = df.groupby("trade_date", group_keys=False)["close"].apply(lambda x: ta.momentum.roc(x, window=12)) #60 min  

    df.drop(columns=["trade_date"], inplace=True)

    df = df.copy()

    return df


"""
Adds volume based indicators (VWAP, CMF) to df

@param df: pd.DataFrame = input DataFrame
@returns pd.DataFrame

"""
def add_volume_indicator(df: pd.DataFrame) -> pd.DataFrame:

    # Isolated Windows
    assert df.index.tz is not None
    df["trade_date"] = df.index.date
    
        #VWAP (Custom)
    df["vwap_3_isolated"] = df.groupby("trade_date", group_keys=False).apply(lambda x: ta.volume.volume_weighted_average_price(x['high'], x['low'], x['close'], x['volume'], window=3)) #15 min
    df["vwap_6_isolated"] = df.groupby("trade_date", group_keys=False).apply(lambda x: ta.volume.volume_weighted_average_price(x['high'], x['low'], x['close'], x['volume'], window=6)) #30 min
    df["vwap_12_isolated"] = df.groupby("trade_date", group_keys=False).apply(lambda x: ta.volume.volume_weighted_average_price(x['high'], x['low'], x['close'], x['volume'], window=12)) #60 min
    
        #CMF (Custom)
    df['cmf_3_isolated'] = df.groupby("trade_date", group_keys=False).apply(lambda x: ta.volume.chaikin_money_flow(x['high'], x['low'], x['close'], x['volume'], window=3)) #15 min
    df['cmf_6_isolated'] = df.groupby("trade_date", group_keys=False).apply(lambda x: ta.volume.chaikin_money_flow(x['high'], x['low'], x['close'], x['volume'], window=6)) #30 min
    df['cmf_12_isolated'] = df.groupby("trade_date", group_keys=False).apply(lambda x: ta.volume.chaikin_money_flow(x['high'], x['low'], x['close'], x['volume'], window=12)) #60 min
    
    df.drop(columns=["trade_date"], inplace=True)
    
    df = df.copy()
    
    return df

"""
Adds volatility based indicators (ATR, BB width) to df

@param df: pd.DataFrame = input DataFrame
@returns pd.DataFrame
"""
def add_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:


    # Isolated Windows
    
    assert df.index.tz is not None
    df["trade_date"] = df.index.date
    
    df['atr_6_isolated'] = df.groupby("trade_date", group_keys=False).apply(lambda x: ta.volatility.average_true_range(x['high'], x['low'], x['close'], window=6)) #30 min
    df['atr_12_isolated'] = df.groupby("trade_date", group_keys=False).apply(lambda x: ta.volatility.average_true_range(x['high'], x['low'], x['close'], window=12)) #60 min
    
    
        #BB (Custom)
    df['bb_width_3_isolated'] = df.groupby("trade_date", group_keys=False).apply(lambda x: ta.volatility.bollinger_wband(x['close'], window=3)) #15 min
    df['bb_width_6_isolated'] = df.groupby("trade_date", group_keys=False).apply(lambda x: ta.volatility.bollinger_wband(x['close'], window=6)) #30 min
    df['bb_width_12_isolated'] = df.groupby("trade_date", group_keys=False).apply(lambda x: ta.volatility.bollinger_wband(x['close'], window=12)) #60 min
   
   
   
    
    df.drop(columns=["trade_date"], inplace=True)
    
    df = df.copy()
    
    return df

"""
Adds trend based indicators (ADX, EMAs) to df

@param df: pd.DataFrame = input DataFrame
@returns pd.DataFrame
"""
def add_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
    
    # Rolling Windows


    # Isolated Windows
    assert df.index.tz is not None
    df["trade_date"] = df.index.date

        # ADX (Custom)
    df["adx_6_isolated"] = df.groupby("trade_date", group_keys=False).apply(lambda x: ta.trend.adx(x['high'], x['low'], x['close'], window=6)) #30 min
    df["adx_12_isolated"] = df.groupby("trade_date", group_keys=False).apply(lambda x: ta.trend.adx(x['high'], x['low'], x['close'], window=12)) #60 min
    
        #EMAs (Custom)
    df['ema_3_isolated'] = df.groupby("trade_date", group_keys=False)["close"].apply(lambda x: ta.trend.ema_indicator(x, window=3)) #15 min
    df['ema_6_isolated'] = df.groupby("trade_date", group_keys=False)["close"].apply(lambda x: ta.trend.ema_indicator(x, window=6)) #30 min

    
    df.drop(columns=["trade_date"], inplace=True)
    
    df = df.copy()
    
    return df


"""
Adds all ta indicators to df
@param df: pd.DataFrame = input DataFrame
@returns pd.DataFrame
"""
def add_ta_indicators(df: pd.DataFrame) -> pd.DataFrame:

    df = add_momentum_indicators(df)
    df = add_volume_indicator(df)
    df = add_volatility_indicators(df)
    df = add_trend_indicators(df)
    
    df = df.copy()
    
    return df

"""
Time (in minutes) till end of trading day (EOD)
@param df: pd.DataFrame = input DataFrame
@returns pd.DataFrame
"""
def add_time_till_eod(df: pd.DataFrame) -> pd.DataFrame:
    df['time_till_eod'] = ((16 - df.index.hour) * 60) - df.index.minute
    return df

"""
Adds approximate sticky volatility regime indicator
@param df: pd.DataFrame - input DataFrame
@returns: pd.DataFrame
"""
def add_vol_regime_approx(df: pd.DataFrame) -> pd.DataFrame:
    
    """
    Garman-Klass Volatility Estimator
    @param df: pd.DataFrame - input DataFrame
    @param window: int - rolling window size
    @param trading_periods: int - number of trading periods in a year
    @returns: pd.Series
    """
    def garman_klass(df, window=30, trading_periods=252) -> pd.Series:

        log_hl = np.log(df["high"] / df["low"])
        log_co = np.log(df["close"] / df["open"])

        rs = 0.5 * log_hl**2 - (2 * math.log(2) - 1) * log_co**2

        result = rs.rolling(window).mean()
        result = np.sqrt(trading_periods * result)

        return result

    
    GK_fast = garman_klass(df, window=12, trading_periods=78*252)
    GK_mid  = garman_klass(df, window=24, trading_periods=78*252)
    GK_slow = garman_klass(df, window=72, trading_periods=78*252)
    

    vol_ok = (GK_fast < GK_mid) & (GK_mid < GK_slow)

    compression_ratio = GK_fast / GK_mid
    compression_ok = (compression_ratio >= 0.85) & (compression_ratio <= 0.95)

    decay_ok = GK_fast.diff() <= 0

    
    # Is a "sticky" volatility regime
    df["is_sticky_vol_regime"] = vol_ok & compression_ok & decay_ok


    return df




"""
Preprocess data
@param df: pd.DataFrame = input DataFrame
@param cols_to_drop: list = list of columns to drop
@param rth: bool = filter to regular trading hours only (True by default)
@returns pd.DataFrame
"""
def preprocess_data(df: pd.DataFrame, cols_to_drop: list = [], rth:bool =True) -> pd.DataFrame:
    

    
    df = add_timestamps(df, rth)
    df = add_ta_indicators(df)

    df = add_time_till_eod(df)
    df = add_vol_regime_approx(df)
    
    
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