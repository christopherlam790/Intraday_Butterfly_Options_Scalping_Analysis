
import pandas as pd
import pyarrow
import numpy as np

import math

def norm_cdf(x):
    """Standard normal CDF (mean=0, std=1)"""
    return 0.5 * (1.0 + math.erf(x / np.sqrt(2.0)))


def get_0_dte_df_options_chain_from_csv(path:str, save_to_parquet:bool = False) -> pd.DataFrame:
    # Load OHLCV
    ohlcv = pd.read_csv(
        path,
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
    
    
    if save_to_parquet:
        df_backtest.to_parquet("assets/clean/spx_ohlcv_1m.parquet")

    
    return df_backtest

    
    

def preprocess_validation(df:pd.DataFrame):
    df = df.copy()

    df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True)
    df["expiration"] = pd.to_datetime(df["expiration"], utc=True)

    df = df.sort_values("ts_event")

    df = df[df["dte"] == 0]
    
    return df



def bs_price_forward(F, K, T, sigma, cp_flag):
    sqrtT = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma * sigma * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    if cp_flag == "C":
        return F * norm_cdf(d1) - K * norm_cdf(d2)
    else:
        return K * norm_cdf(-d2) - F * norm_cdf(-d1)


def implied_vol_bisect(price, F, K, T, cp_flag,
                        lo=1e-4, hi=5.0, tol=1e-6, max_iter=60):
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        val = bs_price_forward(F, K, T, mid, cp_flag)

        if abs(val - price) < tol:
            return mid

        if val > price:
            hi = mid
        else:
            lo = mid

    return np.nan



def bs_delta_forward(F, K, T, sigma, cp_flag):
    d1 = (np.log(F / K) + 0.5 * sigma * sigma * T) / (sigma * np.sqrt(T))
    if cp_flag == "C":
        return norm_cdf(d1)
    else:
        return norm_cdf(d1) - 1.0


def infer_forward(snapshot):
    pcs = snapshot.pivot_table(
        index="strike",
        columns="cp_flag",
        values="close"
    ).dropna()

    F = pcs["C"] - pcs["P"] + pcs.index
    return F.median()



def add_delta(entry_df):
    snap = entry_df.copy()
    F = infer_forward(snap)

    ivs = []
    deltas = []

    for r in snap.itertuples():
        iv = implied_vol_bisect(
            price=r.close,
            F=F,
            K=r.strike,
            T=r.T,
            cp_flag=r.cp_flag
        )

        if np.isnan(iv):
            ivs.append(np.nan)
            deltas.append(np.nan)
        else:
            ivs.append(iv)
            deltas.append(bs_delta_forward(F, r.strike, r.T, iv, r.cp_flag))

    snap["iv"] = ivs
    snap["delta"] = deltas

    return snap.dropna(subset=["delta"])



def atm_by_delta(snapshot, target=0.50):
    calls = snapshot[snapshot.cp_flag == "C"].copy()
    calls["dist"] = np.abs(calls["delta"] - target)
    return calls.loc[calls["dist"].idxmin(), "strike"]


def snapshot_at_time(df, t):
    return df[df["ts_event"].dt.time == t]



def build_butterflies(chain, width=20, cp_flag="C"):
    chain = chain[chain["cp_flag"] == cp_flag]

    strikes = np.sort(chain["strike"].unique())
    butterflies = []

    for k in strikes:
        if k - width in strikes and k + width in strikes:
            lower = chain[chain["strike"] == k - width]
            body  = chain[chain["strike"] == k]
            upper = chain[chain["strike"] == k + width]

            if len(lower) and len(body) and len(upper):
                butterflies.append({
                    "lower_id": lower.iloc[0]["instrument_id"],
                    "body_id": body.iloc[0]["instrument_id"],
                    "upper_id": upper.iloc[0]["instrument_id"],
                    "K": k,
                    "width": width,
                    "cp_flag": cp_flag
                })

    return pd.DataFrame(butterflies)


def price_butterfly(snapshot, fly):
    prices = snapshot.set_index("instrument_id")["close"]

    try:
        debit = (
            prices[fly["lower_id"]]
            - 2 * prices[fly["body_id"]]
            + prices[fly["upper_id"]]
        )
        return debit
    except KeyError:
        return np.nan


if __name__ == "__main__":
        
    
    df = pd.read_parquet("assets/clean/spx_ohlcv_1m.parquet", engine='pyarrow')

    df = preprocess_validation(df)

    df = df.copy()

    df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True)
    df["expiration"] = pd.to_datetime(df["expiration"], utc=True)

    df = df.sort_values("ts_event")

    df = df[df["dte"] == 0]



    ENTRY_TIME = pd.to_datetime("16:00").time()
    EXIT_TIME  = pd.to_datetime("20:00").time()


    df["T"] = (
        (df["expiration"] - df["ts_event"])
        .dt.total_seconds()
        .clip(lower=0)
        / (365 * 24 * 3600)
    )

    df["T"] = df["T"].clip(lower=1e-6)
        
    entry_df = snapshot_at_time(df, ENTRY_TIME)
    entry_df = add_delta(entry_df)        # NumPy-only now
    atm_strike = atm_by_delta(entry_df)



    atm_strike = atm_by_delta(entry_df)

    flies = build_butterflies(
        entry_df,
        width=20,
        cp_flag="C"
    )


    MAX_DIST = 20  # strikes

    flies = flies[
        flies["K"].between(atm_strike - MAX_DIST, atm_strike + MAX_DIST)
    ]


    flies["entry_price"] = flies.apply(
        lambda r: price_butterfly(entry_df, r),
        axis=1
    )


    exit_df = snapshot_at_time(df, EXIT_TIME)

    flies["exit_price"] = flies.apply(
        lambda r: price_butterfly(exit_df, r),
        axis=1
    )

    flies["pnl"] = flies["exit_price"] - flies["entry_price"]


    print(df)

    pass