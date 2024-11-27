import numpy as np
import tensorflow as tf
import random

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Data file paths
SPY_DATA_PATH = '/root/QaFamML/data/spy_data.csv'
ETF_DATA_PATH = '/root/QaFamML/data/etf_adj_close_data.csv'

# Parameters for feature engineering
TAU_EMA = 10  # EMA span parameter for log difference
ALPHA = 2 / (1 + TAU_EMA)  # EMA alpha

# Parameters for Connors RSI (CRSI)
CRSI_F = 3  # Period for Classic RSI
CRSI_G = 2  # Period for Streak RSI
CRSI_H = 3  # Window for Rank RSI

# Portfolio strategy parameters
TAU_ROLLING = 30  # Rolling window size for portfolio strategy
N_ETFs = 12       # Number of ETFs to select in the portfolio

# Autoencoder model parameters
LATENT_DIM = 10
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-3

# Trading days in a year (used for annualizing Sharpe Ratio)
TRADING_DAYS = 252

# Risk-free rate (annualized)
RISK_FREE_RATE = 0.0
