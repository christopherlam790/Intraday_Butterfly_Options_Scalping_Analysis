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
Adds momentum based indicators (RSI, KAMA, ROC) to df

@param df: pd.DataFrame = input DataFrame
@returns: pd.DataFrame
"""
def add_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:

    # Rolling Windows
    
        #RSI (Standard)
    df['rsi_7_rolling'] = ta.momentum.rsi(df['close'], window=7)
    df['rsi_14_rolling'] = ta.momentum.rsi(df['close'], window=14)
    df['rsi_21_rolling'] = ta.momentum.rsi(df['close'], window=21)
    
        #RSI (Custom)
    df['rsi_3_rolling'] = ta.momentum.rsi(df['close'], window=3) #15 min
    df['rsi_6_rolling'] = ta.momentum.rsi(df['close'], window=6) #30 min
    df['rsi_9_rolling'] = ta.momentum.rsi(df['close'], window=9) #45 min
    df['rsi_12_rolling'] = ta.momentum.rsi(df['close'], window=12) #60 min
    
        #KAMA (Standard)
    df['kama_7_rolling'] = ta.momentum.kama(df['close'], window=7)
    df['kama_14_rolling'] = ta.momentum.kama(df['close'], window=14)
    df['kama_21_rolling'] = ta.momentum.kama(df['close'], window=21)
    
        #KAMA (Custom)
    df['kama_3_rolling'] = ta.momentum.kama(df['close'], window=3) #15 min
    df['kama_6_rolling'] = ta.momentum.kama(df['close'], window=6) #30 min
    df['kama_9_rolling'] = ta.momentum.kama(df['close'], window=9) #45 min
    df['kama_12_rolling'] = ta.momentum.kama(df['close'], window=12) #60 min
    
        #ROC (Standard)
    df['roc_7_rolling'] = ta.momentum.roc(df['close'], window=7)
    df['roc_14_rolling'] = ta.momentum.roc(df['close'], window=14)
    df['roc_21_rolling'] = ta.momentum.roc(df['close'], window=21)
    
        #ROC (Custom)
    df['roc_3_rolling'] = ta.momentum.roc(df['close'], window=3) #15 min
    df['roc_6_rolling'] = ta.momentum.roc(df['close'], window=6) #30 min
    df['roc_9_rolling'] = ta.momentum.roc(df['close'], window=9) #45 min
    df['roc_12_rolling'] = ta.momentum.roc(df['close'], window=12) #60 min
    

    # Isolated Windows    
    
    assert df.index.tz is not None
    df["trade_date"] = df.index.date
           
        #RSI (Standard)
    df['rsi_7_isolated'] = df.groupby("trade_date", group_keys=False)["close"].apply(lambda x: ta.momentum.rsi(x, window=7))
    df['rsi_14_isolated'] = df.groupby("trade_date", group_keys=False)["close"].apply(lambda x: ta.momentum.rsi(x, window=14))
    df['rsi_21_isolated'] = df.groupby("trade_date", group_keys=False)["close"].apply(lambda x: ta.momentum.rsi(x, window=21))
    
        #RSI (Custom)
    df['rsi_3_isolated'] = df.groupby("trade_date", group_keys=False)["close"].apply(lambda x: ta.momentum.rsi(x, window=3)) #15 min
    df['rsi_6_isolated'] = df.groupby("trade_date", group_keys=False)["close"].apply(lambda x: ta.momentum.rsi(x, window=6)) #30 min
    df['rsi_9_isolated'] = df.groupby("trade_date", group_keys=False)["close"].apply(lambda x: ta.momentum.rsi(x, window=9)) #45 min
    df['rsi_12_isolated'] = df.groupby("trade_date", group_keys=False)["close"].apply(lambda x: ta.momentum.rsi(x, window=12)) #60 min

        #KAMA (Standard)
    df['kama_7_isolated'] = df.groupby("trade_date", group_keys=False)["close"].apply(lambda x: ta.momentum.kama(x, window=7))
    df['kama_14_isolated'] = df.groupby("trade_date", group_keys=False)["close"].apply(lambda x: ta.momentum.kama(x, window=14))
    df['kama_21_isolated'] = df.groupby("trade_date", group_keys=False)["close"].apply(lambda x: ta.momentum.kama(x, window=21))
    
        #KAMA (Custom)
    df['kama_3_isolated'] = df.groupby("trade_date", group_keys=False)["close"].apply(lambda x: ta.momentum.kama(x, window=3)) #15 min
    df['kama_6_isolated'] = df.groupby("trade_date", group_keys=False)["close"].apply(lambda x: ta.momentum.kama(x, window=6)) #30 min
    df['kama_9_isolated'] = df.groupby("trade_date", group_keys=False)["close"].apply(lambda x: ta.momentum.kama(x, window=9)) #45 min
    df['kama_12_isolated'] = df.groupby("trade_date", group_keys=False)["close"].apply(lambda x: ta.momentum.kama(x, window=12)) #60 min
    
        #ROC (Standard)
    df['roc_7_isolated'] = df.groupby("trade_date", group_keys=False)["close"].apply(lambda x: ta.momentum.roc(x, window=7))
    df['roc_14_isolated'] = df.groupby("trade_date", group_keys=False)["close"].apply(lambda x: ta.momentum.roc(x, window=14))
    df['roc_21_isolated'] = df.groupby("trade_date", group_keys=False)["close"].apply(lambda x: ta.momentum.roc(x, window=21))
    
        #ROC (Custom)
    df['roc_3_isolated'] = df.groupby("trade_date", group_keys=False)["close"].apply(lambda x: ta.momentum.roc(x, window=3)) #15 min
    df['roc_6_isolated'] = df.groupby("trade_date", group_keys=False)["close"].apply(lambda x: ta.momentum.roc(x, window=6)) #30 min
    df['roc_9_isolated'] = df.groupby("trade_date", group_keys=False)["close"].apply(lambda x: ta.momentum.roc(x, window=9)) #45 min
    df['roc_12_isolated'] = df.groupby("trade_date", group_keys=False)["close"].apply(lambda x: ta.momentum.roc(x, window=12)) #60 min  


    df.drop(columns=["trade_date"], inplace=True)

    return df


"""
Adds volume based indicators (VWAP, CMF, OBV) to df

@param df: pd.DataFrame = input DataFrame
@returns pd.DataFrame

"""
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
    
        #CMF (Standard)
    df['cmf_7_rolling'] = ta.volume.chaikin_money_flow(df['high'], df['low'], df['close'], df['volume'], window=7)
    df['cmf_14_rolling'] = ta.volume.chaikin_money_flow(df['high'], df['low'], df['close'], df['volume'], window=14)
    df['cmf_21_rolling'] = ta.volume.chaikin_money_flow(df['high'], df['low'], df['close'], df['volume'], window=21)
    
        #CMF (Custom)
    df['cmf_3_rolling'] = ta.volume.chaikin_money_flow(df['high'], df['low'], df['close'], df['volume'], window=3) #15 min
    df['cmf_6_rolling'] = ta.volume.chaikin_money_flow(df['high'], df['low'], df['close'], df['volume'], window=6) #30 min
    df['cmf_9_rolling'] = ta.volume.chaikin_money_flow(df['high'], df['low'], df['close'], df['volume'], window=9) #45 min
    df['cmf_12_rolling'] = ta.volume.chaikin_money_flow(df['high'], df['low'], df['close'], df['volume'], window=12) #60 min
    
        #OBV (Standard)
    df['obv_rolling'] = ta.volume.on_balance_volume(df['close'], df['volume'])
    
    # Isolated Windows
    assert df.index.tz is not None
    df["trade_date"] = df.index.date
    
        #VWAP (Standard)
    df["vwap_7_isolated"] = df.groupby("trade_date", group_keys=False).apply(lambda x: ta.volume.volume_weighted_average_price(x['high'], x['low'], x['close'], x['volume'], window=7))
    df["vwap_14_isolated"] = df.groupby("trade_date", group_keys=False).apply(lambda x: ta.volume.volume_weighted_average_price(x['high'], x['low'], x['close'], x['volume'], window=14))
    df["vwap_21_isolated"] = df.groupby("trade_date", group_keys=False).apply(lambda x: ta.volume.volume_weighted_average_price(x['high'], x['low'], x['close'], x['volume'], window=21))
    
        #VWAP (Custom)
    df["vwap_3_isolated"] = df.groupby("trade_date", group_keys=False).apply(lambda x: ta.volume.volume_weighted_average_price(x['high'], x['low'], x['close'], x['volume'], window=3)) #15 min
    df["vwap_6_isolated"] = df.groupby("trade_date", group_keys=False).apply(lambda x: ta.volume.volume_weighted_average_price(x['high'], x['low'], x['close'], x['volume'], window=6)) #30 min
    df["vwap_9_isolated"] = df.groupby("trade_date", group_keys=False).apply(lambda x: ta.volume.volume_weighted_average_price(x['high'], x['low'], x['close'], x['volume'], window=9)) #45 min
    df["vwap_12_isolated"] = df.groupby("trade_date", group_keys=False).apply(lambda x: ta.volume.volume_weighted_average_price(x['high'], x['low'], x['close'], x['volume'], window=12)) #60 min
    
        #CMF (Standard)
    df['cmf_7_isolated'] = df.groupby("trade_date", group_keys=False).apply(lambda x: ta.volume.chaikin_money_flow(x['high'], x['low'], x['close'], x['volume'], window=7))
    df['cmf_14_isolated'] = df.groupby("trade_date", group_keys=False).apply(lambda x: ta.volume.chaikin_money_flow(x['high'], x['low'], x['close'], x['volume'], window=14))
    df['cmf_21_isolated'] = df.groupby("trade_date", group_keys=False).apply(lambda x: ta.volume.chaikin_money_flow(x['high'], x['low'], x['close'], x['volume'], window=21))
    
        #CMF (Custom)
    df['cmf_3_isolated'] = df.groupby("trade_date", group_keys=False).apply(lambda x: ta.volume.chaikin_money_flow(x['high'], x['low'], x['close'], x['volume'], window=3)) #15 min
    df['cmf_6_isolated'] = df.groupby("trade_date", group_keys=False).apply(lambda x: ta.volume.chaikin_money_flow(x['high'], x['low'], x['close'], x['volume'], window=6)) #30 min
    df['cmf_9_isolated'] = df.groupby("trade_date", group_keys=False).apply(lambda x: ta.volume.chaikin_money_flow(x['high'], x['low'], x['close'], x['volume'], window=9)) #45 min
    df['cmf_12_isolated'] = df.groupby("trade_date", group_keys=False).apply(lambda x: ta.volume.chaikin_money_flow(x['high'], x['low'], x['close'], x['volume'], window=12)) #60 min
    
        #OBV (Standard)
    df['obv_isolated'] = df.groupby("trade_date", group_keys=False).apply(lambda x: ta.volume.on_balance_volume(x['close'], x['volume']))
    
    df.drop(columns=["trade_date"], inplace=True)
    
    return df

"""
Adds volatility based indicators (ATR, BB, Keltner) to df

@param df: pd.DataFrame = input DataFrame
@returns pd.DataFrame
"""
def add_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
    
    #Rolling Windows
        #ATR (Standard)
    df['atr_7_rolling'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=7)
    df['atr_14_rolling'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    df['atr_21_rolling'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=21)
    
        #ATR (Custom)
    df['atr_3_rolling'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=3) #15 min
    df['atr_6_rolling'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=6) #30 min
    df['atr_9_rolling'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=9) #45 min
    df['atr_12_rolling'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=12) #60 min
    
        #BB (Standard)
    df['bb_width_7_rolling'] = ta.volatility.bollinger_wband(df['close'], window=7)
    df['bb_width_14_rolling'] = ta.volatility.bollinger_wband(df['close'], window=14)
    df['bb_width_21_rolling'] = ta.volatility.bollinger_wband(df['close'], window=21)
    
        #BB (Custom)
    df['bb_width_3_rolling'] = ta.volatility.bollinger_wband(df['close'], window=3) #15 min
    df['bb_width_6_rolling'] = ta.volatility.bollinger_wband(df['close'], window=6) #30 min
    df['bb_width_9_rolling'] = ta.volatility.bollinger_wband(df['close'], window=9) #45 min
    df['bb_width_12_rolling'] = ta.volatility.bollinger_wband(df['close'], window=12) #60 min
    
        #Keltner (Standard)
    df['keltner_width_7_rolling'] = ta.volatility.keltner_channel_wband(df['high'], df['low'], df['close'], window=7)
    df['keltner_width_14_rolling'] = ta.volatility.keltner_channel_wband(df['high'], df['low'], df['close'], window=14)
    df['keltner_width_21_rolling'] = ta.volatility.keltner_channel_wband(df['high'], df['low'], df['close'], window=21)
    
        #Keltner (Custom)
    df['keltner_width_3_rolling'] = ta.volatility.keltner_channel_wband(df['high'], df['low'], df['close'], window=3) #15 min
    df['keltner_width_6_rolling'] = ta.volatility.keltner_channel_wband(df['high'], df['low'], df['close'], window=6) #30 min
    df['keltner_width_9_rolling'] = ta.volatility.keltner_channel_wband(df['high'], df['low'], df['close'], window=9) #45 min
    df['keltner_width_12_rolling'] = ta.volatility.keltner_channel_wband(df['high'], df['low'], df['close'], window=12) #60 min
    
    # Isolated Windows
    
    assert df.index.tz is not None
    df["trade_date"] = df.index.date
    
        #ATR (Standard)
    df['atr_7_isolated'] = df.groupby("trade_date", group_keys=False).apply(lambda x: ta.volatility.average_true_range(x['high'], x['low'], x['close'], window=7))
    df['atr_14_isolated'] = df.groupby("trade_date", group_keys=False).apply(lambda x: ta.volatility.average_true_range(x['high'], x['low'], x['close'], window=14))
    df['atr_21_isolated'] = df.groupby("trade_date", group_keys=False).apply(lambda x: ta.volatility.average_true_range(x['high'], x['low'], x['close'], window=21))
    
        #ATR (Custom)
    df['atr_3_isolated'] = df.groupby("trade_date", group_keys=False).apply(lambda x: ta.volatility.average_true_range(x['high'], x['low'], x['close'], window=3)) #15 min
    df['atr_6_isolated'] = df.groupby("trade_date", group_keys=False).apply(lambda x: ta.volatility.average_true_range(x['high'], x['low'], x['close'], window=6)) #30 min
    df['atr_9_isolated'] = df.groupby("trade_date", group_keys=False).apply(lambda x: ta.volatility.average_true_range(x['high'], x['low'], x['close'], window=9)) #45 min
    df['atr_12_isolated'] = df.groupby("trade_date", group_keys=False).apply(lambda x: ta.volatility.average_true_range(x['high'], x['low'], x['close'], window=12)) #60 min
    
        #BB (Standard)
    df['bb_width_7_isolated'] = df.groupby("trade_date", group_keys=False).apply(lambda x: ta.volatility.bollinger_wband(x['close'], window=7))
    df['bb_width_14_isolated'] = df.groupby("trade_date", group_keys=False).apply(lambda x: ta.volatility.bollinger_wband(x['close'], window=14))
    df['bb_width_21_isolated'] = df.groupby("trade_date", group_keys=False).apply(lambda x: ta.volatility.bollinger_wband(x['close'], window=21))
    
        #BB (Custom)
    df['bb_width_3_isolated'] = df.groupby("trade_date", group_keys=False).apply(lambda x: ta.volatility.bollinger_wband(x['close'], window=3)) #15 min
    df['bb_width_6_isolated'] = df.groupby("trade_date", group_keys=False).apply(lambda x: ta.volatility.bollinger_wband(x['close'], window=6)) #30 min
    df['bb_width_9_isolated'] = df.groupby("trade_date", group_keys=False).apply(lambda x: ta.volatility.bollinger_wband(x['close'], window=9)) #45 min
    df['bb_width_12_isolated'] = df.groupby("trade_date", group_keys=False).apply(lambda x: ta.volatility.bollinger_wband(x['close'], window=12)) #60 min
    
        #Keltner (Standard)
    df['keltner_width_7_isolated'] = df.groupby("trade_date", group_keys=False).apply(lambda x: ta.volatility.keltner_channel_wband(x['high'], x['low'], x['close'], window=7))
    df['keltner_width_14_isolated'] = df.groupby("trade_date", group_keys=False).apply(lambda x: ta.volatility.keltner_channel_wband(x['high'], x['low'], x['close'], window=14))
    df['keltner_width_21_isolated'] = df.groupby("trade_date", group_keys=False).apply(lambda x: ta.volatility.keltner_channel_wband(x['high'], x['low'], x['close'], window=21))
    
        #Keltner (Custom)
    df['keltner_width_3_isolated'] = df.groupby("trade_date", group_keys=False).apply(lambda x: ta.volatility.keltner_channel_wband(x['high'], x['low'], x['close'], window=3)) #15 min
    df['keltner_width_6_isolated'] = df.groupby("trade_date", group_keys=False).apply(lambda x: ta.volatility.keltner_channel_wband(x['high'], x['low'], x['close'], window=6)) #30 min
    df['keltner_width_9_isolated'] = df.groupby("trade_date", group_keys=False).apply(lambda x: ta.volatility.keltner_channel_wband(x['high'], x['low'], x['close'], window=9)) #45 min
    df['keltner_width_12_isolated'] = df.groupby("trade_date", group_keys=False).apply(lambda x: ta.volatility.keltner_channel_wband(x['high'], x['low'], x['close'], window=12)) #60 min
    
    df.drop(columns=["trade_date"], inplace=True)
    
    return df

"""
Adds trend based indicators (ADX, SMAs, EMAs) to df

@param df: pd.DataFrame = input DataFrame
@returns pd.DataFrame
"""
def add_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
    
    # Rolling Windows
        # ADX (Standard)
    df["adx_7_rolling"] = ta.trend.adx(df['high'], df['low'], df['close'], window=7)
    df["adx_14_rolling"] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
    df["adx_21_rolling"] = ta.trend.adx(df['high'], df['low'], df['close'], window=21)
    
        # ADX (Custom)
    df["adx_3_rolling"] = ta.trend.adx(df['high'], df['low'], df['close'], window=3) #15 min
    df["adx_6_rolling"] = ta.trend.adx(df['high'], df['low'], df['close'], window=6) #30 min
    df["adx_9_rolling"] = ta.trend.adx(df['high'], df['low'], df['close'], window=9) #45 min
    df["adx_12_rolling"] = ta.trend.adx(df['high'], df['low'], df['close'], window=12) #60 min
    
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
    
    # Isolated Windows
    assert df.index.tz is not None
    df["trade_date"] = df.index.date
    
        # ADX (Standard)
    df["adx_7_isolated"] = df.groupby("trade_date", group_keys=False).apply(lambda x: ta.trend.adx(x['high'], x['low'], x['close'], window=7))
    df["adx_14_isolated"] = df.groupby("trade_date", group_keys=False).apply(lambda x: ta.trend.adx(x['high'], x['low'], x['close'], window=14))
    df["adx_21_isolated"] = df.groupby("trade_date", group_keys=False).apply(lambda x: ta.trend.adx(x['high'], x['low'], x['close'], window=21))
    
        # ADX (Custom)
    df["adx_3_isolated"] = df.groupby("trade_date", group_keys=False).apply(lambda x: ta.trend.adx(x['high'], x['low'], x['close'], window=3)) #15 min
    df["adx_6_isolated"] = df.groupby("trade_date", group_keys=False).apply(lambda x: ta.trend.adx(x['high'], x['low'], x['close'], window=6)) #30 min
    df["adx_9_isolated"] = df.groupby("trade_date", group_keys=False).apply(lambda x: ta.trend.adx(x['high'], x['low'], x['close'], window=9)) #45 min
    df["adx_12_isolated"] = df.groupby("trade_date", group_keys=False).apply(lambda x: ta.trend.adx(x['high'], x['low'], x['close'], window=12)) #60 min
    
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

    
    df.drop(columns=["trade_date"], inplace=True)
    
    
    return df

def add_ta_indicators(df: pd.DataFrame) -> pd.DataFrame:

    df = add_momentum_indicators(df)
    df = add_volume_indicator(df)
    df = add_volatility_indicators(df)
    df = add_trend_indicators(df)
    
    
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
    
    