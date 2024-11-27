import pandas as pd
import numpy as np

def load_spy_data(spy_data_path):
    spy_df = pd.read_csv(spy_data_path, parse_dates=['Date'])
    spy_df = spy_df.set_index('Date', drop=True)
    spy_df = spy_df.sort_index()
    spy_df['log_returns'] = np.log(spy_df['Adj Close']) - np.log(spy_df['Adj Close'].shift(1))
    spy_df['cum_log_returns'] = spy_df['log_returns'].cumsum()
    spy_df = spy_df[['log_returns', 'cum_log_returns']].fillna(method='bfill')
    return spy_df

def load_etf_data(etf_data_path):
    df = pd.read_csv(etf_data_path, parse_dates=['Date'])
    df = df.set_index('Date', drop=True)
    df = df.sort_index()
    tickers = [c.split('_')[0] for c in df.columns]
    return df, tickers
