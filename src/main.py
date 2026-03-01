import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.layers import Input, Add
from tensorflow.keras.models import Model

from utils import *
from shadowed_rician_channel import *
from models import *

# -----------------------------
# Configuration
# -----------------------------
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
tf.keras.utils.set_random_seed(SEED)

NOMA_SNR = 3
M1, n1, training_SNR1 = 8, 2, 20
M2, n2, training_SNR2 = 4, 2, training_SNR1 - NOMA_SNR
sample_size, N_test, n_epochs, batch_size = 500000, 1000000, 25, 512

oh1 = OneHotEncoder(sparse_output=False, categories=[range(M1)])
oh2 = OneHotEncoder(sparse_output=False, categories=[range(M2)])
train_data1 = oh1.fit_transform(np.random.randint(M1, size=sample_size).reshape(-1,1))
train_data2 = oh2.fit_transform(np.random.randint(M2, size=sample_size).reshape(-1,1))
h_r1, h_i1 = generate_shadowed_rician_H(sample_size)
h_r2, h_i2 = generate_shadowed_rician_H(sample_size)

# -----------------------------
# Build Global Graph
# -----------------------------
input1, input2 = Input(shape=(M1,)), Input(shape=(M2,))
h_r1_in, h_i1_in = Input(shape=(1,)), Input(shape=(1,))
h_r2_in, h_i2_in = Input(shape=(1,)), Input(shape=(1,))

enc1_layer = build_encoder('Encoder1', M1, n1)
enc2_layer = build_encoder('Encoder2', M2, n2)
dec1_layer = build_decoder('Decoder1', M1, n1)
dec2_layer = build_decoder('Decoder2', M2, n2)

f1 = Shadowed_rician_fading_layer()([enc1_layer(input1), h_r1_in, h_i1_in])
f2 = Shadowed_rician_fading_layer()([enc2_layer(input2), h_r2_in, h_i2_in])

ch1 = awgn_channel_layer(SNR_to_noise(training_SNR1, n1))(Add()([f1, f2]))
ch2 = awgn_channel_layer(SNR_to_noise(training_SNR2, n2))(Add()([f2, f1]))

ae1 = Model(inputs=[input1, input2, h_r1_in, h_i1_in, h_r2_in, h_i2_in], outputs=dec1_layer([ch1, h_r1_in, h_i1_in, h_r2_in, h_i2_in]))
ae2 = Model(inputs=[input2, input1, h_r2_in, h_i2_in, h_r1_in, h_i1_in], outputs=dec2_layer([ch2, h_r2_in, h_i2_in, h_r1_in, h_i1_in]))

# -----------------------------
# Training Loop
# -----------------------------
optimizer = keras.optimizers.Nadam(learning_rate=0.006, clipnorm=1.0)
loss_fn = keras.losses.CategoricalCrossentropy()

@tf.function
def train_step(b1, b2, hr1, hi1, hr2, hi2, var_list):
    with tf.GradientTape() as tape:
        p1 = ae1([b1, b2, hr1, hi1, hr2, hi2], training=True)
        p2 = ae2([b2, b1, hr2, hi2, hr1, hi1], training=True)
        loss = tf.reduce_mean(loss_fn(b1, p1)) + tf.reduce_mean(loss_fn(b2, p2))
    grads = tape.gradient(loss, var_list)
    optimizer.apply_gradients(zip(grads, var_list))
    return loss

def train():
    vars = unique_trainable_vars(enc1_layer, enc2_layer, dec1_layer, dec2_layer)
    steps = sample_size // batch_size
    for epoch in range(1, n_epochs + 1):
        loss_val = 0
        for _ in range(steps):
            loss_val += train_step(
                random_batch(train_data1, batch_size), random_batch(train_data2, batch_size),
                random_batch(h_r1, batch_size).reshape(-1,1), random_batch(h_i1, batch_size).reshape(-1,1),
                random_batch(h_r2, batch_size).reshape(-1,1), random_batch(h_i2, batch_size).reshape(-1,1), vars)
        print(f"Epoch {epoch}/{n_epochs}, Loss: {loss_val/steps:.5f}")

if __name__ == "__main__":
    train()