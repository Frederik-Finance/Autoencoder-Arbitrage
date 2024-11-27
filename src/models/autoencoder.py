import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv1D, Flatten, Reshape, MultiHeadAttention
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from configs.config import LATENT_DIM, EPOCHS, BATCH_SIZE, LEARNING_RATE

def prepare_input_matrices(features, tickers):
    d = 3  # Number of features per stock
    input_matrices = []
    for idx, row in features.iterrows():
        r = row[[f'{ticker}_return' for ticker in tickers]].values
        e = row[[f'{ticker}_log_diff_ema' for ticker in tickers]].values
        c = row[[f'{ticker}_CRSI' for ticker in tickers]].values
        sample = np.vstack([r, e, c])
        input_matrices.append(sample)
    X = np.array(input_matrices)
    return X

def build_autoencoder(d, n, latent_dim=10, learning_rate=1e-3):
    input_layer = Input(shape=(d, n))
    x = tf.transpose(input_layer, perm=[0, 2, 1])
    attention = MultiHeadAttention(num_heads=1, key_dim=d)(x, x)
    x = tf.keras.layers.Add()([x, attention])
    x = tf.keras.layers.LayerNormalization()(x)
    conv = Conv1D(filters=latent_dim, kernel_size=1, activation='relu')(x)
    flatten = Flatten()(conv)
    latent = Dense(latent_dim, activation='relu')(flatten)
    reconstruction = Dense(n * latent_dim, activation='relu')(latent)
    reconstruction = Reshape((n, latent_dim))(reconstruction)
    reconstructed_conv = Conv1D(filters=d, kernel_size=1, activation='linear')(reconstruction)
    x = tf.keras.layers.Add()([x, reconstructed_conv])
    output_layer = tf.keras.layers.LayerNormalization()(x)
    output_layer = tf.transpose(output_layer, perm=[0, 2, 1])
    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    optimizer = Adam(learning_rate=learning_rate)
    autoencoder.compile(optimizer=optimizer, loss='mse')
    return autoencoder

def train_autoencoder(X_train, X_test, d, n, latent_dim, epochs, batch_size, learning_rate):
    autoencoder = build_autoencoder(d=d, n=n, latent_dim=latent_dim, learning_rate=learning_rate)
    history = autoencoder.fit(
        X_train, X_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_data=(X_test, X_test)
    )
    return autoencoder, history
