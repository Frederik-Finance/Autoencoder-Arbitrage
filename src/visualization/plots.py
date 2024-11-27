import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_training_history(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
    plt.title('Autoencoder Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_cumulative_returns(comparison_df):
    plt.figure(figsize=(12, 6))
    plt.plot(comparison_df['Portfolio_Cum_Return'], label='Autoencoder Portfolio', color='green')
    plt.plot(comparison_df['Benchmark_Cum_Return'], label='Equally Weighted Benchmark', color='red')
    plt.plot(comparison_df['SPY_Cum_Return'], label='SPY Benchmark', color='lightblue')
    plt.title('Cumulative Returns: Autoencoder Portfolio vs Benchmark')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_reconstruction_errors(stock_errors_series):
    plt.figure(figsize=(12, 6))
    sns.barplot(x=stock_errors_series.index, y=stock_errors_series.values, palette='viridis')
    plt.title('Reconstruction Errors per Stock')
    plt.xlabel('Stocks')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.xticks(rotation=45)
    plt.show()

def plot_correlation_matrix(log_returns, most_uncorrelated_stocks):
    original_corr = log_returns.corr()
    mask = np.triu(np.ones_like(original_corr, dtype=bool))
    plt.figure(figsize=(12, 10))
    sns.heatmap(original_corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Original Correlation Matrix of Log Returns')
    plt.show()
