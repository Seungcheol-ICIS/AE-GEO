{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "\"\"\"Dual GEO Power-Domain Autoencoder framework.\n",
    "    python main.py version\n",
    "\"\"\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "executionInfo": {
     "elapsed": 9830,
     "status": "ok",
     "timestamp": 1754365114612,
     "user": {
      "displayName": "하승철지능미디어공학과",
      "userId": "08871579971026731400"
     },
     "user_tz": -540
    },
    "id": "hVqMu_iSpATH"
   },
   "outputs": [],
   "source": [
    "import os\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import tensorflow as tf\n",
    "from tensorflow import keras\n",
    "from sklearn.preprocessing import OneHotEncoder\n",
    "from tensorflow.keras.layers import (\n",
    "    Input, Dense, Dropout, LayerNormalization, Lambda,\n",
    "    Concatenate, Reshape, Add)\n",
    "from tensorflow.keras.models import Model\n",
    "import pandas as pd\n",
    "import random"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# -----------------------------\n",
    "# Reproducibility\n",
    "# -----------------------------\n",
    "os.environ['PYTHONHASHSEED'] = str(42)\n",
    "tf.config.experimental.enable_op_determinism()\n",
    "SEED = 42\n",
    "random.seed(SEED)\n",
    "np.random.seed(SEED)\n",
    "tf.random.set_seed(SEED)\n",
    "tf.keras.utils.set_random_seed(SEED)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "ybkx2JUAjgRn"
   },
   "source": [
    "**Function**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "executionInfo": {
     "elapsed": 5,
     "status": "ok",
     "timestamp": 1754365114614,
     "user": {
      "displayName": "하승철지능미디어공학과",
      "userId": "08871579971026731400"
     },
     "user_tz": -540
    },
    "id": "7PHT9erBjfNZ"
   },
   "outputs": [],
   "source": [
    "def random_batch(X, batch_size=32):\n",
    "    idx = np.random.randint(len(X), size=batch_size)\n",
    "    return X[idx]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "executionInfo": {
     "elapsed": 4,
     "status": "ok",
     "timestamp": 1754365114616,
     "user": {
      "displayName": "하승철지능미디어공학과",
      "userId": "08871579971026731400"
     },
     "user_tz": -540
    },
    "id": "D5aDGi-B5BeV"
   },
   "outputs": [],
   "source": [
    "def SER_calculator(input_msg, msg):\n",
    "    true_idx = tf.argmax(input_msg, axis=1, output_type=tf.int32)\n",
    "    pred_idx = tf.argmax(msg,       axis=1, output_type=tf.int32)\n",
    "    ser = tf.reduce_mean(tf.cast(tf.not_equal(true_idx, pred_idx), tf.float32))\n",
    "    return ser"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "executionInfo": {
     "elapsed": 12,
     "status": "ok",
     "timestamp": 1754365114630,
     "user": {
      "displayName": "하승철지능미디어공학과",
      "userId": "08871579971026731400"
     },
     "user_tz": -540
    },
    "id": "SHuyL-Ut4-kB"
   },
   "outputs": [],
   "source": [
    "def SNR_to_noise(SNR_db, n=1):\n",
    "    SNR = 10**(SNR_db/10)\n",
    "    noise_std = 1.0 / np.sqrt(SNR * n)\n",
    "    return noise_std"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def complex_mul(a, b):\n",
    "        ar, ai = a[:,0,:], a[:,1,:]\n",
    "        br, bi = b[:,0,:], b[:,1,:]\n",
    "        yr = ar * br - ai * bi\n",
    "        yi = ar * bi + ai * br\n",
    "        return tf.stack([yr, yi], axis=1)\n",
    "\n",
    "def complex_conj(h):\n",
    "    return tf.stack([h[:,0,:], -h[:,1,:]], axis=1)\n",
    "\n",
    "def complex_abs2(h):\n",
    "    return tf.reduce_sum(tf.square(h), axis=1, keepdims=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# -----------------------------\n",
    "# Helper: unique variable list\n",
    "# -----------------------------\n",
    "def unique_trainable_vars(*models_or_layers):\n",
    "    out = []\n",
    "    seen = set()\n",
    "    for m in models_or_layers:\n",
    "        for v in m.trainable_variables:\n",
    "            vid = id(v)\n",
    "            if vid not in seen:\n",
    "                out.append(v)\n",
    "                seen.add(vid)\n",
    "    return out"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Autoencoder Fucntions**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "executionInfo": {
     "elapsed": 15,
     "status": "ok",
     "timestamp": 1754365114660,
     "user": {
      "displayName": "하승철지능미디어공학과",
      "userId": "08871579971026731400"
     },
     "user_tz": -540
    },
    "id": "44QX2oH-K2z7"
   },
   "outputs": [],
   "source": [
    "# -----------------------------\n",
    "# Autoencoder: Encoder\n",
    "# -----------------------------\n",
    "def build_encoder(name_prefix, M, n):\n",
    "    input_signal = Input(shape=(M,), name=f\"{name_prefix}_Input\")\n",
    "    x = Dense(M, activation='relu', name=f\"{name_prefix}_ReLU\")(input_signal)\n",
    "    x = Dense(2 * n, name=f\"{name_prefix}_Dense\")(x)\n",
    "    x = Reshape((2, n), name=f\"{name_prefix}_Reshape\")(x)\n",
    "    # x = Lambda(lambda x: tf.sqrt(tf.cast(n, tf.float32)) * tf.math.l2_normalize(x, axis=[1,2]), name=f\"{name_prefix}_Norm\")(x)\n",
    "    x = Lambda(lambda x: tf.math.l2_normalize(x, axis=[1]), name=f\"{name_prefix}_Norm\")(x)  # Dimension power = 1, test 2\n",
    "    return Model(inputs=input_signal, outputs=x, name=f\"{name_prefix}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "executionInfo": {
     "elapsed": 111,
     "status": "ok",
     "timestamp": 1754365114773,
     "user": {
      "displayName": "하승철지능미디어공학과",
      "userId": "08871579971026731400"
     },
     "user_tz": -540
    },
    "id": "9p-dKIcC-8dO"
   },
   "outputs": [],
   "source": [
    "# -----------------------------\n",
    "# Channel layers\n",
    "# -----------------------------\n",
    "def generate_shadowed_rician_H(batch_size, b=0.126, m=10.1,omega=0.835, r_hat=1.0, phi=0.0): # Average shadowing (AS)\n",
    "    K = omega / (2.0 * b)\n",
    "    p = np.sqrt(K / (1 + K)) * r_hat * np.cos(phi)\n",
    "    q = np.sqrt(K / (1 + K)) * r_hat * np.sin(phi)\n",
    "    sigma = r_hat / np.sqrt(2.0 * (1.0 + K))\n",
    "\n",
    "    xi = np.sqrt(np.random.gamma(shape=m, scale=1.0/m, size=batch_size))\n",
    "    X = np.random.normal(loc=0.0, scale=sigma, size=batch_size) + xi * p\n",
    "    Y = np.random.normal(loc=0.0, scale=sigma, size=batch_size) + xi * q\n",
    "\n",
    "    return X.astype(np.float32), Y.astype(np.float32)\n",
    "\n",
    "def Shadowed_rician_fading_layer(name=None):\n",
    "    def fading(x, h_real, h_imag):\n",
    "        x_real = x[:, 0, :]\n",
    "        x_imag = x[:, 1, :]\n",
    "        y_real = h_real * x_real - h_imag * x_imag\n",
    "        y_imag = h_real * x_imag + h_imag * x_real\n",
    "        return tf.stack([y_real, y_imag], axis=1)\n",
    "    return Lambda(lambda inputs: fading(inputs[0], inputs[1], inputs[2]), name=name)\n",
    "\n",
    "def awgn_channel_layer(noise_std, name=None):\n",
    "    noise_std = tf.convert_to_tensor(noise_std, dtype=tf.float32)\n",
    "    def add_awgn(x):\n",
    "        x_real = x[:, 0, :]\n",
    "        x_imag = x[:, 1, :]\n",
    "        s = noise_std / tf.sqrt(tf.constant(2.0, dtype=tf.float32))\n",
    "        n_real = tf.random.normal(tf.shape(x_real), stddev=s)\n",
    "        n_imag = tf.random.normal(tf.shape(x_imag), stddev=s)\n",
    "        return tf.stack([x_real + n_real, x_imag + n_imag], axis=1)\n",
    "    return Lambda(add_awgn, name=name)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def build_decoder(name_prefix, M, n):\n",
    "    # 입력\n",
    "    y_input   = Input(shape=(2, n), name=\"Rx_Input\")     # 수신 신호\n",
    "    h_real    = Input(shape=(1,),  name=\"h_real\")        # 자기 채널\n",
    "    h_imag    = Input(shape=(1,),  name=\"h_imag\")\n",
    "    h_if_real = Input(shape=(1,),  name=\"h_if_real\")     # 간섭 채널\n",
    "    h_if_imag = Input(shape=(1,),  name=\"h_if_imag\")\n",
    "\n",
    "    y_flat  = Lambda(lambda y: tf.reshape(y, [-1, 2*n]), name=\"Flatten\")(y_input)   # [B, 2n]\n",
    "    h_feat  = Lambda(lambda t: tf.concat(t, axis=1), name=\"Hcat\")([h_real, h_imag, h_if_real, h_if_imag])\n",
    "    x = Lambda(lambda t: tf.concat(t, axis=1), name=\"Merge\")([y_flat, h_feat])\n",
    "\n",
    "    x = keras.layers.Dense(512, activation='relu', name=\"FC1\")(x)\n",
    "    x = keras.layers.LayerNormalization()(x)\n",
    "\n",
    "    x = keras.layers.Dense(256, activation='relu', name=\"FC2\")(x)\n",
    "    x = keras.layers.LayerNormalization()(x)\n",
    "\n",
    "    x = Dense(M, activation='softmax', name=\"Out\")(x)\n",
    "    return keras.Model(inputs=[y_input, h_real, h_imag, h_if_real, h_if_imag], outputs=x, name=f\"{name_prefix}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# -----------------------------\n",
    "# Parameters\n",
    "# -----------------------------\n",
    "NOMA_SNR = 3\n",
    "M1 = 8; n1 = 2; training_SNR1 = 20                         # GEO 1\n",
    "M2 = 4; n2 = 2; training_SNR2 = training_SNR1 - NOMA_SNR   # GEO 2\n",
    "\n",
    "# TRAINING PARAMETERS\n",
    "# sample_size = 300_000; N_test = 500_000; n_epochs = 10; batch_size = 512        # Low epochs\n",
    "sample_size = 500_000; N_test = 1_000_000; n_epochs = 25; batch_size = 512        # Middle epochs\n",
    "# sample_size = 300_000; N_test = 1_000_000; n_epochs = 50; batch_size = 512      # High epochs\n",
    "\n",
    "messages_1 = np.random.randint(M1, size= sample_size)\n",
    "messages_2 = np.random.randint(M2, size= sample_size)\n",
    "\n",
    "h_real_1, h_imag_1 = generate_shadowed_rician_H(sample_size)\n",
    "h_real_2, h_imag_2 = generate_shadowed_rician_H(sample_size)\n",
    "\n",
    "one_hot_encoder1 = OneHotEncoder(categories=[range(M1)], sparse_output=False)\n",
    "one_hot_encoder2 = OneHotEncoder(categories=[range(M2)], sparse_output=False)\n",
    "\n",
    "data_oneH_1 = one_hot_encoder1.fit_transform(messages_1.reshape(-1,1))\n",
    "data_oneH_2 = one_hot_encoder2.fit_transform(messages_2.reshape(-1,1))\n",
    "\n",
    "noise_std1 = SNR_to_noise(training_SNR1, n=n1)\n",
    "noise_std2 = SNR_to_noise(training_SNR2, n=n2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# -----------------------------\n",
    "# Build Autoencoder\n",
    "# -----------------------------\n",
    "# Input\n",
    "input1 = Input(shape=(M1,), name=\"Tx1_Input\")\n",
    "input2 = Input(shape=(M2,), name=\"Tx2_Input\")\n",
    "\n",
    "h_real_input1 = Input(shape=(1,), name=\"H1_real\")\n",
    "h_imag_input1 = Input(shape=(1,), name=\"H1_imag\")\n",
    "h_real_input2 = Input(shape=(1,), name=\"H2_real\")\n",
    "h_imag_input2 = Input(shape=(1,), name=\"H2_imag\")\n",
    "\n",
    "# Encoder\n",
    "encoder1_layer = build_encoder('Encoder1', M1, n1)\n",
    "encoder2_layer = build_encoder('Encoder2', M2, n2)\n",
    "encoder1 = encoder1_layer(input1)\n",
    "encoder2 = encoder2_layer(input2)\n",
    "\n",
    "# Channel\n",
    "faded1 = Shadowed_rician_fading_layer(name=\"Fading1_Layer\")([encoder1, h_real_input1, h_imag_input1])\n",
    "faded2 = Shadowed_rician_fading_layer(name=\"Fading2_Layer\")([encoder2, h_real_input2, h_imag_input2])\n",
    "\n",
    "sum1 = Add(name=\"Sum12\")([faded1, faded2])\n",
    "sum2 = Add(name=\"Sum21\")([faded2, faded1])\n",
    "\n",
    "channel1 = awgn_channel_layer(noise_std1, name=\"AWGN1_layer\")(sum1)\n",
    "channel2 = awgn_channel_layer(noise_std2, name=\"AWGN2_layer\")(sum2)\n",
    "\n",
    "# Decoder\n",
    "decoder1_layer = build_decoder('Decoder1', M1, n1)\n",
    "decoder2_layer = build_decoder('Decoder2', M2, n2)\n",
    "decoder1 = decoder1_layer([channel1, h_real_input1, h_imag_input1, h_real_input2, h_imag_input2])\n",
    "decoder2 = decoder2_layer([channel2, h_real_input2, h_imag_input2, h_real_input1, h_imag_input1])\n",
    "\n",
    "# Autoencoder Models\n",
    "autoencoder1 = Model(\n",
    "    inputs=[input1, input2, h_real_input1, h_imag_input1, h_real_input2, h_imag_input2], \n",
    "    outputs=decoder1, name=\"Autoencoder1\")\n",
    "autoencoder2 = Model(\n",
    "    inputs=[input2, input1, h_real_input2, h_imag_input2, h_real_input1, h_imag_input1],\n",
    "    outputs=decoder2, name=\"Autoencoder2\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# -----------------------------\n",
    "# Plot helpers\n",
    "# -----------------------------\n",
    "def constellation_plot(M1, M2, n1, n2, save_path=None):\n",
    "\n",
    "    inp1 = np.eye(M1, dtype=np.float32)\n",
    "    inp2 = np.eye(M2, dtype=np.float32)\n",
    "\n",
    "    out1 = encoder1_layer(inp1)\n",
    "    out2 = encoder2_layer(inp2)\n",
    "\n",
    "    coding1 = out1.numpy()\n",
    "    coding2 = out2.numpy()\n",
    "\n",
    "    n1_eff = coding1.shape[2]\n",
    "    n2_eff = coding2.shape[2]\n",
    "    max_dim = max(n1_eff, n2_eff)\n",
    "\n",
    "    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']\n",
    "\n",
    "    fig, axes = plt.subplots(1, max_dim + 1, figsize=(5*(max_dim + 1), 5))\n",
    "\n",
    "    if max_dim + 1 == 1:\n",
    "        axes = [axes]\n",
    "\n",
    "    for d in range(max_dim):\n",
    "        ax = axes[d]\n",
    "\n",
    "        if d < n1_eff:\n",
    "            ax.plot(\n",
    "                coding1[:, 0, d], coding1[:, 1, d],\n",
    "                marker=markers[d], markersize=5,\n",
    "                linestyle='None', color='blue',label=f'GEO1 dim{d}'\n",
    "            )\n",
    "\n",
    "        if d < n2_eff:\n",
    "            ax.plot(\n",
    "                coding2[:, 0, d], coding2[:, 1, d],\n",
    "                marker=markers[d], markersize=5,\n",
    "                linestyle='None', color='red', label=f'GEO2 dim{d}'\n",
    "            )\n",
    "\n",
    "        ax.set_title(f'Dimension {d}')\n",
    "        ax.set_xlabel(\"In-Phase\")\n",
    "        ax.set_ylabel(\"Quadrature\")\n",
    "        ax.grid(True)\n",
    "        ax.set_xlim(-2, 2)\n",
    "        ax.set_ylim(-2, 2)\n",
    "        ax.legend()\n",
    "\n",
    "    ax_all = axes[max_dim]\n",
    "\n",
    "    for d in range(max_dim):\n",
    "\n",
    "        if d < n1_eff:\n",
    "            ax_all.plot(\n",
    "                coding1[:, 0, d], coding1[:, 1, d],\n",
    "                marker=markers[d], markersize=3,\n",
    "                linestyle='None', color='blue', label=f'GEO1 dim{d}'\n",
    "            )\n",
    "\n",
    "        if d < n2_eff:\n",
    "            ax_all.plot(\n",
    "                coding2[:, 0, d], coding2[:, 1, d],\n",
    "                marker=markers[d], markersize=3,\n",
    "                linestyle='None', color='red', label=f'GEO2 dim{d}'\n",
    "            )\n",
    "\n",
    "    ax_all.set_title(\"All Dimensions\")\n",
    "    ax_all.set_xlabel(\"In-Phase\")\n",
    "    ax_all.set_ylabel(\"Quadrature\")\n",
    "    ax_all.grid(True)\n",
    "    ax_all.set_xlim(-2, 2)\n",
    "    ax_all.set_ylim(-2, 2)\n",
    "    ax_all.legend(fontsize=8)\n",
    "\n",
    "    plt.tight_layout()\n",
    "\n",
    "    if save_path is not None:\n",
    "        plt.savefig(save_path, dpi=300)\n",
    "\n",
    "    plt.show()\n",
    "\n",
    "\n",
    "\n",
    "def plot_epoch(epoch, n_epochs, epoch_loss, plot_encoding):\n",
    "    el = float(epoch_loss.numpy() if hasattr(epoch_loss, \"numpy\") else epoch_loss)\n",
    "    print(f'Epoch: {epoch}/{n_epochs}, Loss(avg): {el:.5f}')\n",
    "    if plot_encoding:\n",
    "        constellation_plot(M1, M2, n1, n2)\n",
    "\n",
    "\n",
    "def save_constellation_to_csv(encoder1, encoder2, M1, M2, n1, n2, path, tag1, tag2):\n",
    "    inp1 = np.eye(M1, dtype=np.float32)\n",
    "    inp2 = np.eye(M2, dtype=np.float32)\n",
    "\n",
    "    out1 = encoder1(inp1).numpy()  # Shape: (M1, 2, n1)\n",
    "    out2 = encoder2(inp2).numpy()  # Shape: (M2, 2, n2)\n",
    "\n",
    "    data1 = np.transpose(out1, (0, 2, 1)).reshape(M1, -1)\n",
    "    columns1 = [f'dim{i}_{ax}' for i in range(n1) for ax in ['real', 'imag']]\n",
    "    df1 = pd.DataFrame(data1, columns=columns1)\n",
    "    df1['Transmitter'] = 'TX1'\n",
    "    df1['Symbol_Index'] = np.arange(M1)\n",
    "\n",
    "    data2 = np.transpose(out2, (0, 2, 1)).reshape(M2, -1)\n",
    "    columns2 = [f'dim{i}_{ax}' for i in range(n2) for ax in ['real', 'imag']]\n",
    "    df2 = pd.DataFrame(data2, columns=columns2)\n",
    "    df2['Transmitter'] = 'TX2'\n",
    "    df2['Symbol_Index'] = np.arange(M2)\n",
    "\n",
    "    combined_df = pd.concat([df1, df2], ignore_index=True)\n",
    "\n",
    "    id_cols = ['Transmitter', 'Symbol_Index']\n",
    "    data_cols = [col for col in combined_df.columns if col not in id_cols]\n",
    "    combined_df = combined_df[id_cols + data_cols]\n",
    "    \n",
    "    csv_path = os.path.join(path, f\"{tag1}_{tag2}Dim.csv\")\n",
    "    combined_df.to_csv(csv_path, index=False, float_format='%.8f')\n",
    "    print(f\"Saved combined constellation data to: {csv_path}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Simulation and Train Function**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "executionInfo": {
     "elapsed": 24,
     "status": "ok",
     "timestamp": 1754365114861,
     "user": {
      "displayName": "하승철지능미디어공학과",
      "userId": "08871579971026731400"
     },
     "user_tz": -540
    },
    "id": "xCDl-mSVpmQn"
   },
   "outputs": [],
   "source": [
    "# -----------------------------\n",
    "# Train\n",
    "# -----------------------------\n",
    "def train_start(n_epochs=5, n_steps=20, batch_size=100, plot_encoding=True, only_decoder=False,loss_fn=None, optimizer=None):\n",
    "\n",
    "    if only_decoder:\n",
    "        var_list = unique_trainable_vars(decoder1_layer, decoder2_layer)\n",
    "    else:\n",
    "        var_list = unique_trainable_vars(encoder1_layer, encoder2_layer, decoder1_layer, decoder2_layer)\n",
    "\n",
    "    for epoch in range(1, n_epochs + 1):\n",
    "        epoch_mean = keras.metrics.Mean()\n",
    "\n",
    "        for step in range(1, n_steps + 1):\n",
    "            # Batch 데이터 준비\n",
    "            X_batch_1 = tf.convert_to_tensor(random_batch(data_oneH_1, batch_size), dtype=tf.float32)\n",
    "            X_batch_2 = tf.convert_to_tensor(random_batch(data_oneH_2, batch_size), dtype=tf.float32)\n",
    "            h_real_batch_1 = tf.convert_to_tensor(random_batch(h_real_1, batch_size).reshape(-1, 1), dtype=tf.float32)\n",
    "            h_imag_batch_1 = tf.convert_to_tensor(random_batch(h_imag_1, batch_size).reshape(-1, 1), dtype=tf.float32)\n",
    "            h_real_batch_2 = tf.convert_to_tensor(random_batch(h_real_2, batch_size).reshape(-1, 1), dtype=tf.float32)\n",
    "            h_imag_batch_2 = tf.convert_to_tensor(random_batch(h_imag_2, batch_size).reshape(-1, 1), dtype=tf.float32)\n",
    "\n",
    "            with tf.GradientTape() as tape:\n",
    "                y_pred_1 = autoencoder1([X_batch_1, X_batch_2, h_real_batch_1, h_imag_batch_1, h_real_batch_2, h_imag_batch_2], training=True)\n",
    "                y_pred_2 = autoencoder2([X_batch_2, X_batch_1, h_real_batch_2, h_imag_batch_2, h_real_batch_1, h_imag_batch_1], training=True)\n",
    "\n",
    "                # # Cross-entropy loss - Type 1\n",
    "                loss_1 = tf.reduce_mean(loss_fn(X_batch_1, y_pred_1))\n",
    "                loss_2 = tf.reduce_mean(loss_fn(X_batch_2, y_pred_2))\n",
    "                total_loss = loss_1 + loss_2\n",
    "\n",
    "                # # Cross-entropy loss - Type 2\n",
    "                # alpha = 0.5\n",
    "                # loss_1 = tf.reduce_mean(loss_fn(X_batch_1, y_pred_1))\n",
    "                # loss_2 = tf.reduce_mean(loss_fn(X_batch_2, y_pred_2))\n",
    "                # total_loss = alpha * loss_1 + (1.0 - alpha) * loss_2\n",
    "\n",
    "            grads = tape.gradient(total_loss, var_list)\n",
    "            optimizer.apply_gradients(zip(grads, var_list))\n",
    "\n",
    "            epoch_mean.update_state(total_loss)\n",
    "\n",
    "        plot_epoch(epoch, n_epochs, epoch_mean.result(), plot_encoding)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "executionInfo": {
     "elapsed": 13,
     "status": "ok",
     "timestamp": 1754365114848,
     "user": {
      "displayName": "하승철지능미디어공학과",
      "userId": "08871579971026731400"
     },
     "user_tz": -540
    },
    "id": "WxFM3wkKRhb-"
   },
   "outputs": [],
   "source": [
    "# -----------------------------\n",
    "# Test\n",
    "# -----------------------------\n",
    "def Test_AE(data1, data2, SNR_range=None, NOMA_SNR=3):\n",
    "    if SNR_range is None:\n",
    "        SNR_range = np.linspace(0, 20, 11)\n",
    "    SNR_range = np.asarray(SNR_range, dtype=float)\n",
    "    SNR_range1 = SNR_range\n",
    "    SNR_range2 = SNR_range - NOMA_SNR\n",
    "    \n",
    "    SER_transmitter1 = np.empty_like(SNR_range1, dtype=np.float32)\n",
    "    SER_transmitter2 = np.empty_like(SNR_range2, dtype=np.float32)\n",
    "    \n",
    "    H_test_1_real, H_test_1_imag = generate_shadowed_rician_H(len(data1))\n",
    "    H_test_2_real, H_test_2_imag = generate_shadowed_rician_H(len(data2))\n",
    "    \n",
    "    H1r = tf.convert_to_tensor(H_test_1_real.reshape(-1, 1), dtype=tf.float32)\n",
    "    H1i = tf.convert_to_tensor(H_test_1_imag.reshape(-1, 1), dtype=tf.float32)\n",
    "    H2r = tf.convert_to_tensor(H_test_2_real.reshape(-1, 1), dtype=tf.float32)\n",
    "    H2i = tf.convert_to_tensor(H_test_2_imag.reshape(-1, 1), dtype=tf.float32)\n",
    "\n",
    "    encoder1 = encoder1_layer(data1)\n",
    "    encoder2 = encoder2_layer(data2)\n",
    "    \n",
    "    fade_layer1 = Shadowed_rician_fading_layer(name=None)\n",
    "    fade_layer2 = Shadowed_rician_fading_layer(name=None)\n",
    "    faded1 = fade_layer1([encoder1, H1r, H1i])\n",
    "    faded2 = fade_layer2([encoder2, H2r, H2i])\n",
    "    \n",
    "    for i, (snr1, snr2) in enumerate(zip(SNR_range1, SNR_range2)):\n",
    "        sum1 = faded1 + faded2\n",
    "        sum2 = faded2 + faded1\n",
    "        \n",
    "        channel1 = awgn_channel_layer(SNR_to_noise(snr1, n=n1), name=None)(sum1)\n",
    "        channel2 = awgn_channel_layer(SNR_to_noise(snr2, n=n2), name=None)(sum2)\n",
    "        \n",
    "        decoder1 = decoder1_layer([channel1, H1r, H1i, H2r, H2i], training=False)\n",
    "        decoder2 = decoder2_layer([channel2, H2r, H2i, H1r, H1i], training=False)\n",
    "        \n",
    "        SER_transmitter1[i] = SER_calculator(data1, decoder1).numpy()\n",
    "        SER_transmitter2[i] = SER_calculator(data2, decoder2).numpy()\n",
    "        \n",
    "    return (SNR_range1, SER_transmitter1), (SNR_range2, SER_transmitter2)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Train Function**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# -----------------------------\n",
    "# Train setting\n",
    "# -----------------------------\n",
    "test_msg1 = np.random.randint(M1, size=N_test)\n",
    "test_msg2 = np.random.randint(M2, size=N_test)\n",
    "\n",
    "one_hot_encoder1 = OneHotEncoder(sparse_output=False, categories = [range(M1)])\n",
    "one_hot_encoder2 = OneHotEncoder(sparse_output=False, categories = [range(M2)])\n",
    "data_normal1 = one_hot_encoder1.fit_transform(test_msg1.reshape(-1,1))\n",
    "data_normal2 = one_hot_encoder2.fit_transform(test_msg2.reshape(-1,1))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Train Start**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "n_steps = len(data_oneH_1) // batch_size\n",
    "optimizer = keras.optimizers.Nadam(learning_rate=0.006, clipnorm=1.0)\n",
    "loss_fn = keras.losses.CategoricalCrossentropy(label_smoothing=0.0)\n",
    "train_start(n_epochs, n_steps, plot_encoding = True, only_decoder = False, loss_fn=loss_fn, optimizer=optimizer)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "SER_data_transmitter1, SER_data_transmitter2 = Test_AE(\n",
    "    data_normal1, data_normal2, \n",
    "    SNR_range = np.linspace(0, 20, 11), \n",
    "    NOMA_SNR=NOMA_SNR\n",
    "    )"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "2SNg839QmMYe"
   },
   "source": [
    "**Ploting**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 225
    },
    "executionInfo": {
     "elapsed": 190,
     "status": "error",
     "timestamp": 1754365409949,
     "user": {
      "displayName": "하승철지능미디어공학과",
      "userId": "08871579971026731400"
     },
     "user_tz": -540
    },
    "id": "aayz0r8Q_jrX",
    "outputId": "dff99f7e-692f-4c40-f5cf-53c2c2e8c714"
   },
   "outputs": [],
   "source": [
    "# -----------------------------\n",
    "# Plot & Save (Switch version)\n",
    "# -----------------------------\n",
    "\n",
    "# SER performance\n",
    "SNR_db1, SER1_AE = SER_data_transmitter1\n",
    "SNR_db2, SER2_AE = SER_data_transmitter2\n",
    "\n",
    "LW=1.5; MS=5; FS=12\n",
    "\n",
    "plt.figure()\n",
    "plt.semilogy(SNR_db1, SER1_AE, 'b-o', linewidth=LW, markersize=MS, \n",
    "             label=f'[AE] GEO 1 (M={M1}, n={n1})')\n",
    "plt.semilogy(SNR_db1, SER2_AE, 'r-o', linewidth=LW, markersize=MS, \n",
    "             label=f'[AE] GEO 2 (M={M2}, n={n2})')\n",
    "\n",
    "plt.ylim(1e-3, 1)\n",
    "plt.xlim(0, 20)\n",
    "plt.xlabel('SNR [dB]', fontsize=FS)\n",
    "plt.ylabel('SER', fontsize=FS)\n",
    "plt.grid(True, which='both')\n",
    "plt.legend(prop={'size':10}, loc='best')\n",
    "plt.tight_layout()\n",
    "\n",
    "Save = True\n",
    "\n",
    "if Save == True:\n",
    "\n",
    "    ## Save path\n",
    "    path = r'/Users/scha/Library/CloudStorage/Dropbox/ICIS lab/research_24Seungcheol/manuscript-SC/[revision] IEEE Communications Letters (AE design)/revision-20260130/reply_letter/simulation_code/20260218/DataFile'\n",
    "    \n",
    "    tag1 = '20260218_AE'\n",
    "    tag2 = 'OG'\n",
    "\n",
    "    # SER Graph Save\n",
    "    perf_graph_path = os.path.join(path, f\"{tag1}_{tag2}_Graph.png\")\n",
    "    plt.savefig(perf_graph_path, dpi=300)\n",
    "\n",
    "    # Constellation Save\n",
    "    last_fig_path = os.path.join(path, f\"{tag1}_{tag2}_Constellation.png\")\n",
    "    constellation_plot(M1, M2, n1, n2, save_path=last_fig_path)\n",
    "    save_constellation_to_csv(encoder1_layer, encoder2_layer, M1, M2, n1, n2,path, tag1, tag2)\n",
    "\n",
    "    # SER CSV Save\n",
    "    df_raw = pd.DataFrame({\n",
    "        'SER1_AE': SER1_AE,\n",
    "        'SER2_AE': SER2_AE\n",
    "    })\n",
    "\n",
    "    csv_path = os.path.join(path, f\"{tag1}_{tag2}_SER.csv\")\n",
    "    df_raw.round({'SER1_AE': 6, 'SER2_AE': 6}).to_csv(csv_path, index=False)\n",
    "\n",
    "    print(f\"Saved SER csv to: {csv_path}\")\n"
   ]
  }
 ],
 "metadata": {
  "colab": {
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 0
}
