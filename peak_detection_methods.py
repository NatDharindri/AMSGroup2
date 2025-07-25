import os
import sys
import glob
import csv
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, find_peaks_cwt, peak_prominences
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import umap
import torch
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Model, load_model


script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

compute_Threshold_Based_CMD_SPEED = 0
compute_Threshold_Based_TORQUE = 1
compute_Wavelet_Transform = 0
train_model = 0
compute_KMEANS = 0
compute_DBSCAN = 0


# --- Dataset ---

csv_files = glob.glob(r"clean_datasets/*.csv")
df_list = []
for file in csv_files:
    df = pd.read_csv(file, sep=",", encoding="utf-8", parse_dates=True)
    df_list.append(df)

if not csv_files:
    print("No CSV-Files found")


# REMOVE DC-OFFSET
signals_of_interest = ['TORQUE|1', 'TORQUE|2', 'TORQUE|3', 'TORQUE|6', 'CURRENT|1', 'CURRENT|2', 'CURRENT|3', 'CURRENT|6']

for df in df_list:
    for signal_name in signals_of_interest:
        if signal_name in df.columns:
            df[signal_name] = df[signal_name] - df[signal_name].mean()

def convert_decimals(data):
    new_data = []
    for row in data:
        new_row = []
        for item in row:
            if isinstance(item, float):
                item = str(item).replace('.', ',')  # Fließkommazahlen anpassen
            new_row.append(item)
        new_data.append(new_row)
    return new_data


# --- Threshold-Based CMD-SPEED ---
if compute_Threshold_Based_CMD_SPEED:
    csv_data = [
        ["(PAR) Threshold", "(PAR) Minimum Prominence", "Peak Count", "Mean Peak Amplitude", "Mean Peak Prominence", "Number of peakless sequences"]
    ]

    threshold_cmdspeed = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5]
    prominence_cmdspeed = [0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1]

    for param_threshold in threshold_cmdspeed:
        for param_prominence in prominence_cmdspeed:
            peak_counter = 0
            peak_total_height = 0
            peak_mean_height = 0
            peak_total_prominence = 0
            peak_mean_prominence = 0
            samples_without_peak = 0

            for sequence in range(8):
                for axis in [1,2,3,6]:
                    signal = df_list[sequence]['CMD_SPEED|'+str(axis)]
                    target = df_list[sequence]['CURRENT|'+str(axis)]
                    
                    derivative = abs(pd.Series(np.gradient(signal.values, signal.index), index=signal.index))
                    derivative = derivative.clip(lower=0, upper=50)

                    peaks_pos, properties = find_peaks(derivative, height=param_threshold, prominence=param_prominence)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        prominences, _, _ = peak_prominences(target, peaks_pos)
                    
                    if(len(peaks_pos)==0):
                        samples_without_peak = samples_without_peak + 1
                    peak_counter = peak_counter + len(peaks_pos)
                    peak_total_height = peak_total_height + np.sum(abs(target[peaks_pos]))
                    peak_total_prominence = peak_total_prominence + np.sum(prominences)

            if peak_counter > 0:
                peak_mean_height = peak_total_height / peak_counter
                peak_mean_prominence = peak_total_prominence / peak_counter

            new_row = [param_threshold, param_prominence, peak_counter, peak_mean_height, peak_mean_prominence, samples_without_peak]
            csv_data.append(new_row)

    filename = "RESULTS_Threshold-Based-with-CMD-SPEED.csv"

    csv_data = convert_decimals(csv_data)
    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerows(csv_data)
    print("Threshold-Based CMD-SPEED completed")
else:
    print("Skipping Threshold-Based CMD-SPEED")


# --- Threshold-Based TORQUE ---
if compute_Threshold_Based_TORQUE:
    csv_data = [
        ["(PAR) Threshold", "(PAR) Minimum Prominence", "Peak Count", "Mean Peak Amplitude", "Mean Peak Prominence", "Number of peakless sequences"]
    ]

    threshold_torque = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8]
    prominence_torque = [0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1]

    for param_threshold in threshold_torque:
        for param_prominence in prominence_torque:
            peak_counter = 0
            peak_total_height = 0
            peak_mean_height = 0
            peak_total_prominence = 0
            peak_mean_prominence = 0
            samples_without_peak = 0

            for sequence in range(8):
                for axis in [1,2,3,6]:
                    signal = df_list[sequence]['TORQUE|'+str(axis)]
                    target = df_list[sequence]['CURRENT|'+str(axis)]
                
                    peaks_pos, properties = find_peaks(abs(signal), height=param_threshold, prominence=param_prominence)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        prominences, _, _ = peak_prominences(target, peaks_pos)
                    
                    if(len(peaks_pos)==0):
                        samples_without_peak = samples_without_peak + 1
                    peak_counter = peak_counter + len(peaks_pos)
                    peak_total_height = peak_total_height + np.sum(abs(target[peaks_pos]))
                    peak_total_prominence = peak_total_prominence + np.sum(prominences)

            if peak_counter > 0:
                peak_mean_height = peak_total_height / peak_counter
                peak_mean_prominence = peak_total_prominence / peak_counter

            new_row = [param_threshold, param_prominence, peak_counter, peak_mean_height, peak_mean_prominence, samples_without_peak]
            csv_data.append(new_row)

    filename = "RESULTS_Threshold-Based-with-TORQUE.csv"

    csv_data = convert_decimals(csv_data)
    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerows(csv_data)
    print("Threshold-Based TORQUE completed")
else:
    print("Skipping Threshold-Based TORQUE")



# --- Wavelet Transform ---
if compute_Wavelet_Transform:
    csv_data = [
        ["(PAR) Lower Limit", "(PAR) Upper Limit", "Peak Count", "Mean Peak Amplitude", "Mean Peak Prominence", "Number of peakless sequences"]
    ]

    widths = [np.arange(1, 10), np.arange(10, 20), np.arange(20, 40), np.arange(40, 60), np.arange(60, 80), np.arange(80, 100), np.arange(100, 150), np.arange(150, 200)]


    for width in widths:
        peak_counter = 0
        peak_total_height = 0
        peak_mean_height = 0
        peak_total_prominence = 0
        peak_mean_prominence = 0
        samples_without_peak = 0

        for sequence in range(8):
            for axis in [1,2,3,6]:
                signal = df_list[sequence]['TORQUE|'+str(axis)]
                target = df_list[sequence]['CURRENT|'+str(axis)]
                
                peaks_pos = find_peaks_cwt(signal, width) # Ricker-Wavelet
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    prominences, _, _ = peak_prominences(target, peaks_pos)
                    
                if(len(peaks_pos)==0):
                    samples_without_peak = samples_without_peak + 1
                peak_counter = peak_counter + len(peaks_pos)
                peak_total_height = peak_total_height + np.sum(abs(target[peaks_pos]))
                peak_total_prominence = peak_total_prominence + np.sum(prominences)

        if peak_counter > 0:
            peak_mean_height = peak_total_height / peak_counter
            peak_mean_prominence = peak_total_prominence / peak_counter

        new_row = [width.min(), width.max(), peak_counter, peak_mean_height, peak_mean_prominence, samples_without_peak]
        csv_data.append(new_row)

    filename = "RESULTS_Wavelet-Transform.csv"

    csv_data = convert_decimals(csv_data)
    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerows(csv_data)
    print("Wavelet Transform completed")
else:
    print("Skipping Wavelet Transform")


# --- Clustering ---
def extract_multichannel_windows(signal, window_size=50, stride=10):
    signals = StandardScaler().fit_transform(signal.values.reshape(-1, 1)).flatten()
        
    windows = []
    for start in range(0, len(signals) - window_size + 1, stride):
        window = signals[start:start + window_size]
        windows.append(window)
    return np.array(windows)  # shape: [n_windows, window_size, n_features]

def label_time_series(labels, stride, window_size, total_length):
    time_labels = np.full(total_length, -1)
    for i, label in enumerate(labels):
        start = i * stride
        end = min(start + window_size, total_length)
        time_labels[start:end] = label
    return time_labels

feature = 'TORQUE|'  # TORQUE     CMD_SPEED

if (compute_KMEANS or compute_DBSCAN):
    X_all = []
    for sequence in range(8):
        for axis in [1, 2, 3, 6]:
            signal = df_list[sequence][feature+str(axis)]
            windows = extract_multichannel_windows(signal, window_size=50, stride=10)
            X_all.append(windows)

    X = np.vstack(X_all)

    # === Autoencoder ===
    FEATURE_SIZE = 1
    WINDOW_SIZE = 50
    STRIDE = 10
    LATENT_DIM = 32
    input_shape = (WINDOW_SIZE, FEATURE_SIZE) #(WINDOW_SIZE, len(FEATURES))

if (compute_KMEANS or compute_DBSCAN) and train_model:
    # === Encoder ===
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv1D(32, 3, activation='relu', padding='same')(inputs)  # 50
    x = layers.MaxPooling1D(2, padding='same')(x)  # 50 → 25
    x = layers.Conv1D(16, 3, activation='relu', padding='same')(x)  # 25
    x = layers.MaxPooling1D(2, padding='same')(x)  # 25 → 13
    x = layers.Flatten()(x)
    latent = layers.Dense(LATENT_DIM, name='latent')(x)

    # === Decoder ===
    x = layers.Dense(13 * 16)(latent)
    x = layers.Reshape((13, 16))(x)  # 13 time steps
    x = layers.UpSampling1D(2)(x)  # 13 → 26
    x = layers.Conv1D(16, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling1D(2)(x)  # 26 → 52
    x = layers.Conv1D(32, 3, activation='relu', padding='same')(x)
    x = layers.Conv1D(FEATURE_SIZE, 3, activation='linear', padding='same')(x)  # 52
    x = layers.Cropping1D((1, 1))(x)  # 52 → 50

    outputs = x

    autoencoder = models.Model(inputs, outputs)
    autoencoder.compile(optimizer='adam', loss='mse')
    # autoencoder.summary()

    encoder = Model(inputs=autoencoder.input,
                    outputs=autoencoder.get_layer('latent').output)

    # === Autoencoder Training ===
    autoencoder.fit(
        X, X,
        epochs=30,
        batch_size=64,
        validation_split=0.1,
        shuffle=True
    )
    if(feature == 'CMD_SPEED|'):
        autoencoder.save("models/autoencoder_CMD_SPEED.keras")
        encoder.save("models/encoder_CMD_SPEED.keras")
    if(feature == 'TORQUE|'):
        autoencoder.save("models/autoencoder_TORQUE.keras")
        encoder.save("models/encoder_TORQUE.keras")


# --- KMEANS ---

if compute_KMEANS:
    csv_data = [
        ["(PAR) Feature", "Peak Count", "Mean Peak Amplitude", "Mean Peak Prominence", "Number of peakless sequences"]
    ]

    features = ['CMD_SPEED', 'TORQUE']

    for f in features:
        kmeans = KMeans(n_clusters=2, random_state=0) # 2-Cluster-KMeans for peak vs. non-peak
        if(f == 'CMD_SPEED'):
            autoencoder = load_model("models/autoencoder_CMD_SPEED.keras")
            encoder = load_model("models/encoder_CMD_SPEED.keras")
        elif(f == 'TORQUE' and not train_model):
            autoencoder = load_model("models/autoencoder_TORQUE.keras")
            encoder = load_model("models/encoder_TORQUE.keras")


        peak_counter = 0
        peak_total_height = 0
        peak_mean_height = 0
        peak_total_prominence = 0
        peak_mean_prominence = 0
        samples_without_peak = 0

        for sequence in range(8):
            for axis in [1,2,3,6]:
                signal = df_list[sequence][f+"|"+str(axis)]
                target = df_list[sequence]['CURRENT|'+str(axis)]

                X = extract_multichannel_windows(signal, window_size=50, stride=10)
                latent_vectors = encoder.predict(X)
                labels_kmeans = kmeans.fit_predict(latent_vectors)

                total_length = len(signal)
                labels_for_signal = labels_kmeans[:(total_length - WINDOW_SIZE) // STRIDE + 1]
                cluster_series = label_time_series(labels_for_signal, STRIDE, WINDOW_SIZE, total_length)
                peak_label = 1
                mask = (cluster_series == peak_label)

                peaks_pos = np.where(mask)[0]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    prominences, _, _ = peak_prominences(target, peaks_pos)  
                    
                if(len(peaks_pos)==0):
                    samples_without_peak = samples_without_peak + 1
                peak_counter = peak_counter + len(peaks_pos)
                peak_total_height = peak_total_height + np.sum(abs(target[peaks_pos]))
                peak_total_prominence = peak_total_prominence + np.sum(prominences)

        if peak_counter > 0:
            peak_mean_height = peak_total_height / peak_counter
            peak_mean_prominence = peak_total_prominence / peak_counter

        new_row = [f, peak_counter, peak_mean_height, peak_mean_prominence, samples_without_peak]
        csv_data.append(new_row)

    filename = "RESULTS_KMEANS.csv"

    csv_data = convert_decimals(csv_data)
    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerows(csv_data)
    print("KMEANS completed")
else:
    print("Skipping KMEANS")


# --- DBSCAN ---
if compute_DBSCAN:
    csv_data = [
        ["(PAR) Feature", "(PAR) Epsilon", "Peak Count", "Mean Peak Amplitude", "Mean Peak Prominence", "Number of peakless sequences"]
    ]

    eps_values = [0.5, 1, 1.5, 2, 2.5, 3, 3.5]
    features = ['CMD_SPEED', 'TORQUE']

    for f in features:
        if(f == 'CMD_SPEED'):
            autoencoder = load_model("models/autoencoder_CMD_SPEED.keras")
            encoder = load_model("models/encoder_CMD_SPEED.keras")
        elif(f == 'TORQUE'):
            autoencoder = load_model("models/autoencoder_TORQUE.keras")
            encoder = load_model("models/encoder_TORQUE.keras")

        for eps in eps_values:
            kmeans = KMeans(n_clusters=2, random_state=0) # 2-Cluster-KMeans for „peak“ vs. non-peak“

            peak_counter = 0
            peak_total_height = 0
            peak_mean_height = 0
            peak_total_prominence = 0
            peak_mean_prominence = 0
            samples_without_peak = 0

            for sequence in range(8):
                for axis in [1,2,3,6]:
                    signal = df_list[sequence][f+'|'+str(axis)]
                    target = df_list[sequence]['CURRENT|'+str(axis)]

                    X = extract_multichannel_windows(signal, window_size=50, stride=10)
                    latent_vectors = encoder.predict(X)
                    clustering = DBSCAN(eps=eps, min_samples=5).fit(latent_vectors)
                    labels_dbscan = clustering.labels_

                    total_length = len(signal)
                    labels_for_signal = labels_dbscan[:(total_length - WINDOW_SIZE) // STRIDE + 1]
                    cluster_series = label_time_series(labels_for_signal, STRIDE, WINDOW_SIZE, total_length)
                    peak_label = -1
                    mask = (cluster_series == peak_label)

                    peaks_pos = np.where(mask)[0]
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        prominences, _, _ = peak_prominences(target, peaks_pos)  
                        
                    if(len(peaks_pos)==0):
                        samples_without_peak = samples_without_peak + 1
                    peak_counter = peak_counter + len(peaks_pos)
                    peak_total_height = peak_total_height + np.sum(abs(target[peaks_pos]))
                    peak_total_prominence = peak_total_prominence + np.sum(prominences)

            if peak_counter > 0:
                peak_mean_height = peak_total_height / peak_counter
                peak_mean_prominence = peak_total_prominence / peak_counter

            new_row = [f, eps, peak_counter, peak_mean_height, peak_mean_prominence, samples_without_peak]
            csv_data.append(new_row)

    filename = "RESULTS_DBSCAN.csv"

    csv_data = convert_decimals(csv_data)
    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerows(csv_data)
    print("DBSCAN completed")
else:
    print("Skipping DBSCAN")