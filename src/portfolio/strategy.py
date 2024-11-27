import pandas as pd
import numpy as np
from configs.config import TAU_ROLLING, N_ETFs

def reconstruct_returns(autoencoder, X_test, features, tickers):
    reconstructed_X_test = autoencoder.predict(X_test)
    reconstructed_returns = reconstructed_X_test[:, 0, :]
    reconstructed_returns_df = pd.DataFrame(
        reconstructed_returns,
        index=features.iloc[-X_test.shape[0]:].index,
        columns=tickers
    )
    actual_returns_df = pd.DataFrame(
        X_test[:, 0, :],
        index=features.iloc[-X_test.shape[0]:].index,
        columns=tickers
    )
    return reconstructed_returns_df, actual_returns_df

def compute_portfolio_returns(reconstructed_returns_df, actual_returns_df, spy_df, tickers, TAU, N_ETFs):
    diff_returns = reconstructed_returns_df - actual_returns_df
    rolling_diff = diff_returns.rolling(window=TAU).mean()
    rolling_diff = rolling_diff.dropna()
    portfolio_returns = []
    selected_stocks = []
    for current_time in rolling_diff.index:
        p_tau_t = rolling_diff.loc[current_time]
        selected = p_tau_t.sort_values().head(N_ETFs).index.tolist()
        selected_stocks.append(selected)
        portfolio_return = actual_returns_df.loc[current_time, selected].mean()
        portfolio_returns.append(portfolio_return)
    portfolio_df = pd.DataFrame({
        'Portfolio_Return': portfolio_returns
    }, index=rolling_diff.index)
    benchmark_returns = actual_returns_df.loc[rolling_diff.index].mean(axis=1)
    benchmark_df = pd.DataFrame({
        'Benchmark_Return': benchmark_returns
    }, index=rolling_diff.index)
    comparison_df = pd.concat([portfolio_df, benchmark_df], axis=1)
    comparison_df['Portfolio_Cum_Return'] = (1 + comparison_df['Portfolio_Return']).cumprod() - 1
    comparison_df['Benchmark_Cum_Return'] = (1 + comparison_df['Benchmark_Return']).cumprod() - 1
    comparison_df['SPY_Return'] = spy_df['log_returns'].loc[comparison_df.index]
    comparison_df['SPY_Cum_Return'] = (spy_df['log_returns'].loc[comparison_df.index]).cumsum()
    return comparison_df, selected_stocks
