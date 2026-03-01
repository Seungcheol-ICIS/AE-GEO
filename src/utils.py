import tensorflow as tf
import numpy as np
import pandas as pd
import os

def random_batch(X, batch_size=32):
    idx = np.random.randint(len(X), size=batch_size)
    return X[idx]

def SER_calculator(input_msg, msg):
    true_idx = tf.argmax(input_msg, axis=1, output_type=tf.int32)
    pred_idx = tf.argmax(msg, axis=1, output_type=tf.int32)
    ser = tf.reduce_mean(tf.cast(tf.not_equal(true_idx, pred_idx), tf.float32))
    return ser

def SNR_to_noise(SNR_db, n=1):
    SNR = 10**(SNR_db/10)
    noise_std = 1.0 / np.sqrt(SNR * n)
    return noise_std

def unique_trainable_vars(*models_or_layers):
    out = []
    seen = set()
    for m in models_or_layers:
        for v in m.trainable_variables:
            vid = id(v)
            if vid not in seen:
                out.append(v)
                seen.add(vid)
    return out

def save_constellation_to_csv(encoder1, encoder2, M1, M2, n1, n2, path, tag1, tag2):
    inp1 = np.eye(M1, dtype=np.float32)
    inp2 = np.eye(M2, dtype=np.float32)
    out1 = encoder1(inp1).numpy()
    out2 = encoder2(inp2).numpy()

    data1 = np.transpose(out1, (0, 2, 1)).reshape(M1, -1)
    df1 = pd.DataFrame(data1, columns=[f'dim{i}_{ax}' for i in range(n1) for ax in ['real', 'imag']])
    df1['Transmitter'], df1['Symbol_Index'] = 'TX1', np.arange(M1)

    data2 = np.transpose(out2, (0, 2, 1)).reshape(M2, -1)
    df2 = pd.DataFrame(data2, columns=[f'dim{i}_{ax}' for i in range(n2) for ax in ['real', 'imag']])
    df2['Transmitter'], df2['Symbol_Index'] = 'TX2', np.arange(M2)

    combined_df = pd.concat([df1, df2], ignore_index=True)
    csv_path = os.path.join(path, f"{tag1}_{tag2}Dim.csv")
    combined_df.to_csv(csv_path, index=False, float_format='%.8f')
    print(f"Saved constellation data to: {csv_path}")