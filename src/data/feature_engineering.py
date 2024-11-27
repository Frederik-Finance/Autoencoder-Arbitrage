import pandas as pd
import numpy as np
from configs.config import TAU_EMA, CRSI_F, CRSI_G, CRSI_H

def compute_log_returns(df, tickers):
    log_returns = np.log(df) - np.log(df.shift(1))
    log_returns = log_returns.dropna()
    log_returns.columns = [f'{ticker}_return' for ticker in tickers]
    return log_returns

def compute_log_diff_ema(df, tickers, tau):
    ema = df.ewm(span=tau, adjust=False).mean()
    log_diff_ema = np.log(df) - np.log(ema)
    log_diff_ema = log_diff_ema.dropna()
    log_diff_ema.columns = [f'{ticker}_log_diff_ema' for ticker in tickers]
    return log_diff_ema

def compute_classic_rsi(series, period=3):
    delta = series.diff(1)
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_streak(series):
    streak = series.copy()
    streak[streak > 0] = 1
    streak[streak < 0] = -1
    streak = streak.fillna(0)
    return streak

def compute_streak_rsi(series, period=2):
    streak = compute_streak(series)
    streak_sum = streak.rolling(window=period, min_periods=1).sum()
    streak_sum = streak_sum.apply(lambda x: 1 if abs(x) > period else x)
    delta = streak_sum.diff(1)
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_rank_rsi(series, window=3):
    rank = series.rolling(window=window, min_periods=1).apply(lambda x: (x > x.iloc[-1]).sum() / len(x), raw=False)
    return rank * 100

def compute_crsi(log_returns, tickers, f=3, g=2, h=3):
    classic_rsi = pd.DataFrame(index=log_returns.index)
    streak_rsi = pd.DataFrame(index=log_returns.index)
    rank_rsi = pd.DataFrame(index=log_returns.index)

    for ticker in tickers:
        rsi_f = compute_classic_rsi(log_returns[f'{ticker}_return'], period=f)
        classic_rsi[f'{ticker}_CRSI_F'] = rsi_f

        rsi_g = compute_streak_rsi(log_returns[f'{ticker}_return'], period=g)
        streak_rsi[f'{ticker}_CRSI_G'] = rsi_g

        rsi_h = compute_rank_rsi(log_returns[f'{ticker}_return'], window=h)
        rank_rsi[f'{ticker}_CRSI_H'] = rsi_h

    crsi_grouped = {}
    for ticker in tickers:
        f_col = f'{ticker}_CRSI_F'
        g_col = f'{ticker}_CRSI_G'
        h_col = f'{ticker}_CRSI_H'
        if f_col in classic_rsi.columns and g_col in streak_rsi.columns and h_col in rank_rsi.columns:
            crsi_grouped[ticker] = (classic_rsi[f_col] + streak_rsi[g_col] + rank_rsi[h_col]) / 3
        else:
            raise ValueError(f"Missing CRSI components for ticker {ticker}")

    crsi = pd.DataFrame(crsi_grouped)
    crsi = crsi * 1e-3
    crsi.columns = [f'{ticker}_CRSI' for ticker in tickers]
    return crsi

def assemble_features(log_returns, log_diff_ema, crsi):
    common_index = log_returns.index.intersection(log_diff_ema.index).intersection(crsi.index)
    log_returns = log_returns.loc[common_index]
    log_diff_ema = log_diff_ema.loc[common_index]
    crsi = crsi.loc[common_index]
    features = pd.concat([log_returns, log_diff_ema, crsi], axis=1)
    features = features.dropna()
    return features
