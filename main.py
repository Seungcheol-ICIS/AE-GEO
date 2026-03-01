import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.layers import Input, Add
from tensorflow.keras.models import Model

# 사용자 정의 모듈 임포트
from utils import *
from shadowed_rician_channel import *
from models import *

def get_args():
    parser = argparse.ArgumentParser(description="AE-NOMA Training and Simulation")
    
    # Training Hyperparameters
    parser.add_argument('--num-samples', type=int, default=500000, help='Number of training samples')
    parser.add_argument('--num-epochs', type=int, default=25, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.006, help='Learning rate')
    
    # Communication Parameters
    parser.add_argument('--noma-snr', type=int, default=3, help='SNR offset for NOMA (dB)')
    parser.add_argument('--training-snr', type=int, default=20, help='Base training SNR (dB)')
    parser.add_argument('--n-test', type=int, default=1000000, help='Number of test samples')
    
    # Save/Path Options
    parser.add_argument('--tag', type=str, default='AE_NOMA_GEO', help='Tag for saving files')
    parser.add_argument('--save-path', type=str, default='./results', help='Directory to save results')
    
    return parser.parse_args()

@tf.function
def train_step(b1, b2, hr1, hi1, hr2, hi2, var_list, ae1, ae2, loss_fn, optimizer):
    with tf.GradientTape() as tape:
        p1 = ae1([b1, b2, hr1, hi1, hr2, hi2], training=True)
        p2 = ae2([b2, b1, hr2, hi2, hr1, hi1], training=True)
        loss = tf.reduce_mean(loss_fn(b1, p1)) + tf.reduce_mean(loss_fn(b2, p2))
    grads = tape.gradient(loss, var_list)
    optimizer.apply_gradients(zip(grads, var_list))
    return loss

def main():
    args = get_args()
    
    # 고정 파라미터 및 SEED 설정
    SEED = 42
    tf.keras.utils.set_random_seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    
    M1, n1 = 8, 2
    M2, n2 = 4, 2
    training_SNR2 = args.training_snr - args.noma_snr

    # 1. 데이터 준비
    oh1 = OneHotEncoder(sparse_output=False, categories=[range(M1)])
    oh2 = OneHotEncoder(sparse_output=False, categories=[range(M2)])
    
    train_data1 = oh1.fit_transform(np.random.randint(M1, size=args.num_samples).reshape(-1,1))
    train_data2 = oh2.fit_transform(np.random.randint(M2, size=args.num_samples).reshape(-1,1))
    h_r1, h_i1 = generate_shadowed_rician_H(args.num_samples)
    h_r2, h_i2 = generate_shadowed_rician_H(args.num_samples)

    # 2. 모델 구축
    input1, input2 = Input(shape=(M1,)), Input(shape=(M2,))
    h_r1_in, h_i1_in = Input(shape=(1,)), Input(shape=(1,))
    h_r2_in, h_i2_in = Input(shape=(1,)), Input(shape=(1,))

    enc1 = build_encoder('Encoder1', M1, n1)
    enc2 = build_encoder('Encoder2', M2, n2)
    dec1 = build_decoder('Decoder1', M1, n1)
    dec2 = build_decoder('Decoder2', M2, n2)

    f1 = Shadowed_rician_fading_layer()([enc1(input1), h_r1_in, h_i1_in])
    f2 = Shadowed_rician_fading_layer()([enc2(input2), h_r2_in, h_i2_in])

    ch1 = awgn_channel_layer(SNR_to_noise(args.training_snr, n1))(Add()([f1, f2]))
    ch2 = awgn_channel_layer(SNR_to_noise(training_SNR2, n2))(Add()([f2, f1]))

    ae1 = Model(inputs=[input1, input2, h_r1_in, h_i1_in, h_r2_in, h_i2_in], outputs=dec1([ch1, h_r1_in, h_i1_in, h_r2_in, h_i2_in]))
    ae2 = Model(inputs=[input2, input1, h_r2_in, h_i2_in, h_r1_in, h_i1_in], outputs=dec2([ch2, h_r2_in, h_i2_in, h_r1_in, h_i1_in]))

    # 3. 학습 루프
    optimizer = keras.optimizers.Nadam(learning_rate=args.lr, clipnorm=1.0)
    loss_fn = keras.losses.CategoricalCrossentropy()
    vars_to_train = unique_trainable_vars(enc1, enc2, dec1, dec2)
    
    print(f"\n--- Training Start (Epochs: {args.num_epochs}, Batch: {args.batch_size}) ---")
    steps = args.num_samples // args.batch_size
    for epoch in range(1, args.num_epochs + 1):
        total_loss = 0
        for _ in range(steps):
            total_loss += train_step(
                random_batch(train_data1, args.batch_size), random_batch(train_data2, args.batch_size),
                random_batch(h_r1, args.batch_size).reshape(-1,1), random_batch(h_i1, args.batch_size).reshape(-1,1),
                random_batch(h_r2, args.batch_size).reshape(-1,1), random_batch(h_i2, args.batch_size).reshape(-1,1),
                vars_to_train, ae1, ae2, loss_fn, optimizer)
        print(f"Epoch {epoch}/{args.num_epochs}, Loss: {total_loss/steps:.5f}")

    # 4. 테스트 (SER 평가)
    print("\n--- Testing Start ---")
    test_msg1 = np.random.randint(M1, size=args.n_test)
    test_msg2 = np.random.randint(M2, size=args.n_test)
    data_test1 = oh1.transform(test_msg1.reshape(-1,1)).astype(np.float32)
    data_test2 = oh2.transform(test_msg2.reshape(-1,1)).astype(np.float32)

    H_t1r, H_t1i = generate_shadowed_rician_H(args.n_test)
    H_t2r, H_t2i = generate_shadowed_rician_H(args.n_test)
    t_hr1, t_hi1 = H_t1r.reshape(-1, 1), H_t1i.reshape(-1, 1)
    t_hr2, t_hi2 = H_t2r.reshape(-1, 1), H_t2i.reshape(-1, 1)

    faded1 = Shadowed_rician_fading_layer()([enc1(data_test1), t_hr1, t_hi1])
    faded2 = Shadowed_rician_fading_layer()([enc2(data_test2), t_hr2, t_hi2])
    sig_sum = faded1 + faded2

    snr_range = np.linspace(0, 20, 11)
    ser1_list, ser2_list = [], []

    for snr in snr_range:
        rx1 = awgn_channel_layer(SNR_to_noise(snr, n1))(sig_sum)
        rx2 = awgn_channel_layer(SNR_to_noise(snr - args.noma_snr, n2))(sig_sum)
        
        pred1 = dec1([rx1, t_hr1, t_hi1, t_hr2, t_hi2], training=False)
        pred2 = dec2([rx2, t_hr2, t_hi2, t_hr1, t_hi1], training=False)
        
        ser1_list.append(SER_calculator(data_test1, pred1).numpy())
        ser2_list.append(SER_calculator(data_test2, pred2).numpy())
        print(f"SNR: {snr}dB | GEO1 SER: {ser1_list[-1]:.5f} | GEO2 SER: {ser2_list[-1]:.5f}")

    # 5. 결과 저장 및 시각화
    if not os.path.exists(args.save_path): os.makedirs(args.save_path)
    
    # 그래프 출력
    plt.figure(figsize=(8, 6))
    plt.semilogy(snr_range, ser1_list, 'b-o', label=f'GEO1 (M={M1})')
    plt.semilogy(snr_range, ser2_list, 'r-s', label=f'GEO2 (M={M2}, Offset={args.noma_snr}dB)')
    plt.grid(True, which='both')
    plt.xlabel('SNR [dB]')
    plt.ylabel('SER')
    plt.title(f'AE-NOMA (Tag: {args.tag})')
    plt.legend()
    plt.savefig(os.path.join(args.save_path, f"{args.tag}_SER.png"))
    
    # 데이터 저장
    save_constellation_to_csv(enc1, enc2, M1, M2, n1, n2, args.save_path, args.tag, "Final")
    pd.DataFrame({'SNR': snr_range, 'SER1': ser1_list, 'SER2': ser2_list}).to_csv(
        os.path.join(args.save_path, f"{args.tag}_Results.csv"), index=False)
    
    print(f"\nAll results saved to {args.save_path}")

if __name__ == "__main__":
    main()
