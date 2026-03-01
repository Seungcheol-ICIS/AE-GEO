import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Reshape, Lambda, LayerNormalization, Concatenate
from tensorflow.keras.models import Model

def build_encoder(name_prefix, M, n):
    input_signal = Input(shape=(M,), name=f"{name_prefix}_Input")
    x = Dense(M, activation='relu')(input_signal)
    x = Dense(2 * n)(x)
    x = Reshape((2, n))(x)
    x = Lambda(lambda x: tf.math.l2_normalize(x, axis=[1]), name=f"{name_prefix}_Norm")(x)
    return Model(inputs=input_signal, outputs=x, name=name_prefix)

def build_decoder(name_prefix, M, n):
    y_input = Input(shape=(2, n), name="Rx_Input")
    h_r = Input(shape=(1,), name="h_real")
    h_i = Input(shape=(1,), name="h_imag")
    h_if_r = Input(shape=(1,), name="h_if_real")
    h_if_i = Input(shape=(1,), name="h_if_imag")

    y_flat = Reshape((2*n,))(y_input)
    h_feat = Concatenate()([h_r, h_i, h_if_r, h_if_i])
    x = Concatenate()([y_flat, h_feat])

    x = Dense(512, activation='relu')(x)
    x = LayerNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = LayerNormalization()(x)
    
    output = Dense(M, activation='softmax')(x)
    return Model(inputs=[y_input, h_r, h_i, h_if_r, h_if_i], outputs=output, name=name_prefix)