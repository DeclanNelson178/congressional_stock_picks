import yfinance as yf
import os
from tqdm import tqdm
import pandas as pdk
import pickle
import hashlib
from pathlib import Path

import pandas as pd
import hashlib

def hash_dataframe(df, hash_type="md5"):
    """
    Produce a hash string from a pandas DataFrame.
    
    Args:
        df (pd.DataFrame): The DataFrame to hash
        hash_type (str): Hash algorithm ('md5', 'sha256', etc.)
    
    Returns:
        str: Hash string
    """
    # Sort DataFrame to ensure consistent hash
    df_sorted = df.sort_index(axis=0).sort_index(axis=1)

    # Convert to bytes
    data_bytes = pd.util.hash_pandas_object(df_sorted, index=True).values

    # Create the hash
    hash_func = hashlib.new(hash_type)
    hash_func.update(data_bytes)

    return hash_func.hexdigest()

# Set a cache directory
CACHE_DIR = "/Users/declannelson/Desktop/code/congressional_stock_picks/data/returns_cache"

# Make sure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

def format_date(date):
    return f"{date:%Y-%m-%d}"

def get_cache_filename(ticker, start, end):
    # Create a simple unique ID based on arguments
    id_string = f"{ticker}_{format_date(start)}_{format_date(end)}"
    safe_id = id_string.replace(" ", "_").replace(":", "-")
    filename = f"{safe_id}.parquet"
    return os.path.join(CACHE_DIR, filename)

def fetch_stock_data(ticker, start, end, silent=False):
    """
    Fetch stock data for a given ticker and time range.
    Check if cached locally, otherwise download and cache.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    cache_file = get_cache_filename(ticker, start or "None", end or "None")

    if os.path.exists(cache_file):
        if not silent: 
            print(f"✅ Loading cached data: {cache_file}")
        return pd.read_parquet(cache_file)
    
    if not silent: 
        print(f"⬇️  Downloading data for {ticker} from Yahoo Finance...")

    data = yf.download(ticker, start=format_date(start), end=format_date(end))
    
    if data.empty:
        raise ValueError(f"No data found for {ticker} between {start} and {end}")

    data.to_parquet(cache_file)
    if not silent:
        print(f"💾 Cached data to {cache_file}")
    
    return data


def cache_all_stock_data(df):
    cache_dir = Path(CACHE_DIR) / hash_dataframe(df)
    failure_cache_dir = Path(CACHE_DIR) / hash_dataframe(df) / "failed_tickers"
    os.makedirs(failure_cache_dir, exist_ok=True)
    if (cache_dir / "df_beta_adj.parquet").exists():
        print(f"Data is cached -- reading {cache_dir / 'df_beta_adj.parquet'}")
        df = pd.read_parquet(cache_dir / "df_beta_adj.parquet")
        print("reading failed tickers")
        with open(cache_dir / "failed_tickers.pkl", "rb") as file:
            failed_tickers= pickle.load(file)
    else:
        print("Stock data not cached -- running now")
        dates = df["transaction_date"].unique()
        start_date, end_date = min(dates), max(dates)
        start_date -= pd.Timedelta(days=30)
        end_date += pd.Timedelta(days=30)
        dfs = []
        failed_tickers = []
        tickers = ["SPY", *df["ticker"].unique()]

        for ticker in tqdm(tickers):
            ticker_file = ticker + ".txt"
            if (failure_cache_dir / ticker_file).exists(): 
                print("Found failed marker")
                failed_tickers.append(ticker)
                continue

            try:
                df = fetch_stock_data(ticker, start_date, end_date, silent=True)
                df = df.stack("Ticker")
                df.index.names = ["date", "ticker"]
                dfs.append(df)
            except ValueError:
                (failure_cache_dir / f"{ticker}.txt").touch()
                print(f"Writing {failure_cache_dir / ticker}.txt")
                failed_tickers.append(ticker)
        
        df = pd.concat(dfs)
        df.to_parquet(cache_dir / "df_beta_adj.parquet", index=True)
        with open(cache_dir / "failed_tickers.pkl", "wb") as file:
            pickle.dump(failed_tickers, file)

    return df, failed_tickers


def construct_forward_backward_returns_frame(price_df):
    # Calculate daily returns first
    price_df = price_df.sort_index()
    price_df = price_df["Close"]
    group = price_df.groupby("ticker")
    dfs = []

    for days in (1, 5, 15, 30):
        dfs.append(group.apply(lambda x: (x.shift(-1 * days) / x) - 1).to_frame(f"returns_after_{days}d").join(
            group.apply(lambda x: (x / x.shift(days) ) - 1).to_frame(f"returns_before_{days}d")
        ))
    
    return pd.concat(dfs, axis=1).droplevel(0)

def construct_forward_backward_returns_adtv(price_df):
    # Calculate daily returns first
    price_df = price_df.sort_index()
    price_df = price_df["Volume"]
    dfs = []

    for days in (1, 5, 15, 30):
        dfs.append(price_df.shift(days).rolling(window=days, min_periods=1).mean().to_frame(f"adtv_before_{days}d"))
        dfs.append(price_df.shift(-days).rolling(window=days, min_periods=1).mean().to_frame(f"adtv_after_{days}d"))
    dfs.append(price_df.to_frame("daily_adtv"))
    return pd.concat(dfs, axis=1)