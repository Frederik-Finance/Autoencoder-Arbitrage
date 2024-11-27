from configs.config import (
    SPY_DATA_PATH, ETF_DATA_PATH, TAU_EMA, CRSI_F, CRSI_G, CRSI_H,
    LATENT_DIM, EPOCHS, BATCH_SIZE, LEARNING_RATE, TAU_ROLLING,
    N_ETFs, TRADING_DAYS, RISK_FREE_RATE
)
from src.data.data_loader import load_spy_data, load_etf_data
from src.data.feature_engineering import (
    compute_log_returns, compute_log_diff_ema, compute_crsi, assemble_features
)
from src.models.autoencoder import prepare_input_matrices, train_autoencoder
from src.portfolio.strategy import reconstruct_returns, compute_portfolio_returns
from src.visualization.plots import (
    plot_training_history, plot_cumulative_returns,
    plot_reconstruction_errors, plot_correlation_matrix
)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def main():
    # Load data
    spy_df = load_spy_data(SPY_DATA_PATH)
    df, tickers = load_etf_data(ETF_DATA_PATH)
    
    # Feature Engineering
    log_returns = compute_log_returns(df, tickers)
    log_diff_ema = compute_log_diff_ema(df, tickers, TAU_EMA)
    crsi = compute_crsi(log_returns, tickers, CRSI_F, CRSI_G, CRSI_H)
    features = assemble_features(log_returns, log_diff_ema, crsi)
    
    # Prepare data for autoencoder
    X = prepare_input_matrices(features, tickers)
    d = 3  # Number of features per stock
    n = len(tickers)
    
    # Train-test split
    X_train, X_test = train_test_split(X, test_size=0.2, shuffle=False)
    
    # Train autoencoder
    autoencoder, history = train_autoencoder(
        X_train, X_test, d, n,
        latent_dim=LATENT_DIM, epochs=EPOCHS,
        batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate model
    test_loss = autoencoder.evaluate(X_test, X_test)
    print(f'\nTest Loss (MSE): {test_loss}')
    
    # Reconstruct returns and compute portfolio returns
    reconstructed_returns_df, actual_returns_df = reconstruct_returns(autoencoder, X_test, features, tickers)
    comparison_df, selected_stocks = compute_portfolio_returns(
        reconstructed_returns_df, actual_returns_df, spy_df, tickers, TAU_ROLLING, N_ETFs
    )
    
    # Plot cumulative returns
    plot_cumulative_returns(comparison_df)
    
    # Calculate Sharpe Ratios
    daily_risk_free_rate = RISK_FREE_RATE / TRADING_DAYS
    comparison_df['Portfolio_Excess_Return'] = comparison_df['Portfolio_Return'] - daily_risk_free_rate
    comparison_df['Benchmark_Excess_Return'] = comparison_df['Benchmark_Return'] - daily_risk_free_rate
    comparison_df['SPY_Excess_Return'] = comparison_df['SPY_Return'] - daily_risk_free_rate
    
    mean_portfolio = comparison_df['Portfolio_Excess_Return'].mean()
    std_portfolio = comparison_df['Portfolio_Excess_Return'].std()
    mean_benchmark = comparison_df['Benchmark_Excess_Return'].mean()
    std_benchmark = comparison_df['Benchmark_Excess_Return'].std()
    mean_spy = comparison_df['SPY_Excess_Return'].mean()
    std_spy = comparison_df['SPY_Excess_Return'].std()
    
    sharpe_portfolio = (mean_portfolio * TRADING_DAYS) / (std_portfolio * np.sqrt(TRADING_DAYS))
    sharpe_benchmark = (mean_benchmark * TRADING_DAYS) / (std_benchmark * np.sqrt(TRADING_DAYS))
    sharpe_spy = (mean_spy * TRADING_DAYS) / (std_spy * np.sqrt(TRADING_DAYS))
    
    sharpe_df = pd.DataFrame({
        'Sharpe_Ratio': [sharpe_portfolio, sharpe_benchmark, sharpe_spy]
    }, index=['Autoencoder Portfolio', 'Equally Weighted Benchmark', 'SPY Benchmark'])
    
    print("\nSharpe Ratios:")
    print(sharpe_df)
    
    # Compute reconstruction errors
    reconstructed_X_test = autoencoder.predict(X_test)
    reconstruction_errors = ((X_test - reconstructed_X_test) ** 2).mean(axis=0)
    stock_errors = reconstruction_errors.sum(axis=0)
    stock_errors_series = pd.Series(stock_errors, index=tickers)
    
    print("\nReconstruction Errors per Stock:")
    print(stock_errors_series)
    
    sorted_errors = stock_errors_series.sort_values(ascending=False)
    print("\nStocks Sorted by Reconstruction Error (High to Low):")
    print(sorted_errors)
    
    N = 5  # Number of top uncorrelated stocks
    most_uncorrelated_stocks = sorted_errors.head(N).index.tolist()
    print(f"\nTop {N} Most Uncorrelated Stocks Based on Reconstruction Error:")
    print(most_uncorrelated_stocks)
    
    # Plot reconstruction errors
    plot_reconstruction_errors(stock_errors_series)
    
    # Plot correlation matrix
    plot_correlation_matrix(log_returns, most_uncorrelated_stocks)

if __name__ == '__main__':
    main()
