import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.layers import Input, Add
from tensorflow.keras.models import Model
import pandas as pd

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
print("--- Training Start ---")
train()

print("\n--- Testing Start ---")
test_msg1 = np.random.randint(M1, size=N_test)
test_msg2 = np.random.randint(M2, size=N_test)

data_normal1 = oh1.transform(test_msg1.reshape(-1,1)).astype(np.float32)
data_normal2 = oh2.transform(test_msg2.reshape(-1,1)).astype(np.float32)
snr_range = np.linspace(0, 20, 11)

ser1_list = []
ser2_list = []

H_t1r, H_t1i = generate_shadowed_rician_H(N_test)
H_t2r, H_t2i = generate_shadowed_rician_H(N_test)

t_hr1 = tf.convert_to_tensor(H_t1r.reshape(-1, 1))
t_hi1 = tf.convert_to_tensor(H_t1i.reshape(-1, 1))
t_hr2 = tf.convert_to_tensor(H_t2r.reshape(-1, 1))
t_hi2 = tf.convert_to_tensor(H_t2i.reshape(-1, 1))

enc1_out = enc1_layer(data_normal1)
enc2_out = enc2_layer(data_normal2)

faded1 = Shadowed_rician_fading_layer()([enc1_out, t_hr1, t_hi1])
faded2 = Shadowed_rician_fading_layer()([enc2_out, t_hr2, t_hi2])
sig_sum = faded1 + faded2

for snr in snr_range:
    snr1 = snr
    snr2 = snr - NOMA_SNR
    
    rx1 = awgn_channel_layer(SNR_to_noise(snr1, n1))(sig_sum)
    rx2 = awgn_channel_layer(SNR_to_noise(snr2, n2))(sig_sum)
    
    pred1 = dec1_layer([rx1, t_hr1, t_hi1, t_hr2, t_hi2], training=False)
    pred2 = dec2_layer([rx2, t_hr2, t_hi2, t_hr1, t_hi1], training=False)
    
    ser1 = SER_calculator(data_normal1, pred1)
    ser2 = SER_calculator(data_normal2, pred2)
    
    ser1_list.append(ser1.numpy())
    ser2_list.append(ser2.numpy())
    print(f"SNR: {snr}dB | GEO1 SER: {ser1:.5f} | GEO2 SER: {ser2:.5f}")

print("\n--- Saving Results ---")

save_path = './results' 
if not os.path.exists(save_path): os.makedirs(save_path)

tag = " "

plt.figure(figsize=(8, 6))
plt.semilogy(snr_range, ser1_list, 'b-o', label=f'GEO1 (M={M1})')
plt.semilogy(snr_range, ser2_list, 'r-s', label=f'GEO2 (M={M2}, Offset={NOMA_SNR}dB)')
plt.grid(True, which='both')
plt.xlabel('SNR [dB]')
plt.ylabel('Symbol Error Rate (SER)')
plt.title('AE-NOMA Performance in Shadowed Rician Channel')
plt.legend()
plt.savefig(os.path.join(save_path, f"{tag}_SER_Curve.png"))
plt.show()

save_constellation_to_csv(enc1_layer, enc2_layer, M1, M2, n1, n2, save_path, tag, "Final")

res_df = pd.DataFrame({'SNR': snr_range, 'SER1': ser1_list, 'SER2': ser2_list})
res_df.to_csv(os.path.join(save_path, f"{tag}_Results.csv"), index=False)

print(f"All results saved to {save_path}")
