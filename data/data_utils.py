from pathlib import Path
import pandas as pd
from data.stock_returns import cache_all_stock_data, construct_forward_backward_returns_adtv, construct_forward_backward_returns_frame, fetch_stock_data
from enum import Enum, auto
import yfinance as yf
import numpy as np
from sklearn.preprocessing import StandardScaler

FEATURE_PATH = "/Users/declannelson/Desktop/code/congressional_stock_picks/data/feature_beta_adj_df.parquet"


def load_raw_congressional_transactions():
    return pd.read_csv("/Users/declannelson/Desktop/code/congressional_stock_picks/data/all_transactions.csv")

def load_raw_stock_transactions():
    df = load_raw_congressional_transactions()
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    return df.loc[df["asset_type"] == "Stock"]

def amount_map(amount):
    min_amount, max_amount = amount.strip().replace(" ", "").replace("$", "").replace(",", "").split("-")
    return (int(min_amount) + int(max_amount)) / 2


def load_feature_set():
    path = Path(FEATURE_PATH)
    if path.exists():
        return pd.read_parquet(path).dropna()
    
    df = load_raw_stock_transactions()
    price_df = cache_all_stock_data(df)[0]
    total_tickers = df["ticker"].nunique()

    df = load_raw_stock_transactions()
    price_df = cache_all_stock_data(df)[0]

    spy_price = price_df.xs("SPY", level="ticker")[["Close"]]
    raw_price_df = price_df.loc[~price_df.index.isin(["SPY"], level="ticker"), ["Close"]]
    raw_price_df = raw_price_df.join(spy_price, rsuffix="_spy", on="date", how="left").drop_duplicates()
    rolling_cov = raw_price_df["Close"].rolling(window=60).cov(raw_price_df["Close_spy"])
    rolling_var = raw_price_df["Close_spy"].rolling(window=60).var()
    rolling_beta = rolling_cov / rolling_var
    raw_price_df["Close"] = raw_price_df["Close"] - (rolling_beta * raw_price_df["Close_spy"])
    price_df = price_df.drop(columns=["Close"]).join(raw_price_df["Close"], how="left")

    ret_df = construct_forward_backward_returns_frame(price_df)
    adtv_df = construct_forward_backward_returns_adtv(price_df)
    df = df.rename(columns={"transaction_date": "date"}).set_index(["date", "ticker"])

    valid_tickers = ret_df.index.unique("ticker")
    print(f"({len(valid_tickers)} / {total_tickers}) tickers are valid")

    df = df.loc[df.index.isin(valid_tickers, level=1)]
    df["partial"] = df["type"] == "Sale (Partial)"
    df["amount"] = df["amount"].apply(amount_map)
    df["direction"] = df["type"].str.contains("Sale").apply(lambda sale: -1 if sale else 1)
    df["amount"] = df["direction"] * df["amount"]
    df = df.loc[df["type"] != "Exchange"]

    df = df.groupby(["date", "senator", "ticker"]).sum().reset_index(["senator"])
    df["direction"] = df["amount"] / df["amount"].abs()
    df["amount"] = df["amount"].abs()

    df = df.join(ret_df).join(adtv_df).reset_index().set_index(["date", "senator", "ticker"])

    df = df[["amount", "direction", *ret_df.columns, *adtv_df.columns]].sort_index().dropna()
    df.to_parquet(path)
    return df

def load_standardized_feature_set(include_known: bool):
    df = load_feature_set()
    ret_columns = [c for c in df.columns if c.startswith("returns_")]
    vol_columns = [c for c in df.columns if c.startswith("adtv_") or c == "daily_adtv"]

    features_to_scale = ret_columns + vol_columns + ['amount']
    scaler = StandardScaler()
    df_scaled_features = scaler.fit_transform(df[features_to_scale])
    df_scaled = pd.DataFrame(df_scaled_features, columns=features_to_scale, index=df.index)
    df_scaled["direction"] = df["direction"]

    if include_known:
        df_scaled["known_instance"] = 0
        df_scaled.loc[pd.IndexSlice[pd.Timestamp(2020, 1, 24):pd.Timestamp(2020, 2, 14), "Kelly Loeffler", :], "known_instance"] = 1
        df_scaled.loc[pd.IndexSlice[pd.Timestamp(2020, 1, 24):pd.Timestamp(2020, 3, 2), "David Perdue", :], "known_instance"] = 1
        df_scaled.loc[pd.IndexSlice[pd.Timestamp(2020, 1, 27):pd.Timestamp(2020, 1, 27), "James M. Inhofe", :], "known_instance"] = 1
        df_scaled.loc[pd.IndexSlice[pd.Timestamp(2018, 1, 1):pd.Timestamp(2020, 1, 1), "David Perdue", "BWXT"], "known_instance"] = 1

    return df_scaled


