import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Lambda

def generate_shadowed_rician_H(batch_size, b=0.126, m=10.1, omega=0.835, r_hat=1.0, phi=0.0):
    K = omega / (2.0 * b)
    sigma = r_hat / np.sqrt(2.0 * (1.0 + K))
    p = np.sqrt(K / (1 + K)) * r_hat * np.cos(phi)
    q = np.sqrt(K / (1 + K)) * r_hat * np.sin(phi)

    xi = np.sqrt(np.random.gamma(shape=m, scale=1.0/m, size=batch_size))
    X = np.random.normal(loc=0.0, scale=sigma, size=batch_size) + xi * p
    Y = np.random.normal(loc=0.0, scale=sigma, size=batch_size) + xi * q
    return X.astype(np.float32), Y.astype(np.float32)

def Shadowed_rician_fading_layer(name=None):
    def fading(inputs):
        x, h_real, h_imag = inputs
        x_real, x_imag = x[:, 0, :], x[:, 1, :]
        y_real = h_real * x_real - h_imag * x_imag
        y_imag = h_real * x_imag + h_imag * x_real
        return tf.stack([y_real, y_imag], axis=1)
    return Lambda(fading, name=name)

def awgn_channel_layer(noise_std, name=None):
    noise_std = tf.convert_to_tensor(noise_std, dtype=tf.float32)
    def add_awgn(x):
        s = noise_std / tf.sqrt(2.0)
        noise = tf.random.normal(tf.shape(x), stddev=s)
        return x + noise
    return Lambda(add_awgn, name=name)