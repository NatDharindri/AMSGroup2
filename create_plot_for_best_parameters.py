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
compute_Threshold_Based_TORQUE = 0
compute_Wavelet_Transform = 0
compute_KMEANS = 0
compute_DBSCAN = 0


# --- Dataset ---

csv_files = sorted(glob.glob(r"clean_datasets/*.csv"))
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
    param_threshold = 0.7
    param_prominence = 1.1

    save_dir = "plots/RESULTS_Threshold-Based-with-CMD-SPEED"
    os.makedirs(save_dir, exist_ok=True)

    for sequence in range(8):
        for axis in [1,2,3,6]:
            signal = df_list[sequence]['CMD_SPEED|'+str(axis)]
            target = df_list[sequence]['CURRENT|'+str(axis)]
                    
            derivative = abs(pd.Series(np.gradient(signal.values, signal.index), index=signal.index))
            derivative = derivative.clip(lower=0, upper=50)

            peaks_pos, properties = find_peaks(derivative, height=param_threshold, prominence=param_prominence)

            plt.figure(figsize=(10, 4))
            plt.plot(target, label='Current-Signal')
            plt.plot(peaks_pos, target.iloc[peaks_pos], "rx", label='Peaks')
            plt.grid(True)
            plt.legend(fontsize=12)
            plt.tick_params(axis='both', labelsize=12)
            plt.savefig(f"{save_dir}/sequence{sequence}_axis{axis}.png", dpi=300, bbox_inches='tight')
            # plt.show()
            plt.close()

    
    print("Threshold-Based CMD-SPEED completed")
else:
    print("Skipping Threshold-Based CMD-SPEED")


# --- Threshold-Based TORQUE ---
if compute_Threshold_Based_TORQUE:
    param_threshold = 5
    param_prominence = 1.1

    save_dir = "plots/RESULTS_Threshold-Based-with-TORQUE"
    os.makedirs(save_dir, exist_ok=True)

    for sequence in range(8):
        for axis in [1,2,3,6]:
            signal = df_list[sequence]['TORQUE|'+str(axis)]
            target = df_list[sequence]['CURRENT|'+str(axis)]
                
            peaks_pos, properties = find_peaks(abs(signal), height=param_threshold, prominence=param_prominence) #, width=30, distance=50)

            plt.figure(figsize=(10, 4))
            plt.plot(target, label='Current-Signal')
            plt.plot(peaks_pos, target.iloc[peaks_pos], "rx", label='Peaks')
            plt.grid(True)
            plt.legend(fontsize=12)
            plt.tick_params(axis='both', labelsize=12)
            plt.savefig(f"{save_dir}/sequence{sequence}_axis{axis}.png", dpi=300, bbox_inches='tight')
            # plt.show()
            plt.close()

    
    print("Threshold-Based TORQUE completed")
else:
    print("Skipping Threshold-Based TORQUE")



# --- Wavelet Transform ---
if compute_Wavelet_Transform:
    width = np.arange(100, 150)

    save_dir = "plots/RESULTS_Wavelet-Transform"
    os.makedirs(save_dir, exist_ok=True)

    for sequence in range(8):
        for axis in [1,2,3,6]:
            signal = df_list[sequence]['TORQUE|'+str(axis)]
            target = df_list[sequence]['CURRENT|'+str(axis)]
                
            peaks_pos = find_peaks_cwt(signal, width) # Ricker-Wavelet
            
            plt.figure(figsize=(10, 4))
            plt.plot(target, label='Current-Signal')
            plt.plot(peaks_pos, target.iloc[peaks_pos], "rx", label='Peaks')
            plt.grid(True)
            plt.legend(fontsize=12)
            plt.tick_params(axis='both', labelsize=12)
            plt.savefig(f"{save_dir}/sequence{sequence}_axis{axis}.png", dpi=300, bbox_inches='tight')
            # plt.show()
            plt.close()
    

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

if (compute_KMEANS or compute_DBSCAN):
    # === Autoencoder ===
    FEATURE_SIZE = 1
    WINDOW_SIZE = 50
    STRIDE = 10
    LATENT_DIM = 32
    input_shape = (WINDOW_SIZE, FEATURE_SIZE) #(WINDOW_SIZE, len(FEATURES))




# --- KMEANS ---

if compute_KMEANS:
    feature = 'CMD_SPEED'

    save_dir = "plots/RESULTS_KMEANS"
    os.makedirs(save_dir, exist_ok=True)

    kmeans = KMeans(n_clusters=2, random_state=0) # 2-Cluster-KMeans for „peak“ vs. non-peak“
    if(feature == 'CMD_SPEED'):
        autoencoder = load_model("models/autoencoder_CMD_SPEED.keras")
        encoder = load_model("models/encoder_CMD_SPEED.keras")
    elif(feature == 'TORQUE' and not train_model):
        autoencoder = load_model("models/autoencoder_TORQUE.keras")
        encoder = load_model("models/encoder_TORQUE.keras")

    for sequence in range(8):
        for axis in [1,2,3,6]:
            signal = df_list[sequence][feature+"|"+str(axis)]
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

            plt.figure(figsize=(10, 4))
            plt.plot(target, label='Current-Signal')
            plt.plot(peaks_pos, target.iloc[peaks_pos], "rx", label='Peaks')
            plt.grid(True)
            plt.legend(fontsize=12)
            plt.tick_params(axis='both', labelsize=12)
            plt.savefig(f"{save_dir}/sequence{sequence}_axis{axis}.png", dpi=300, bbox_inches='tight')
            # plt.show()
            plt.close()
                

    print("KMEANS completed")
else:
    print("Skipping KMEANS")


# --- DBSCAN ---
if compute_DBSCAN:
    eps = 3.5
    feature = 'TORQUE'

    save_dir = "plots/RESULTS_DBSCAN"
    os.makedirs(save_dir, exist_ok=True)

    if(feature == 'CMD_SPEED'):
        autoencoder = load_model("models/autoencoder_CMD_SPEED.keras")
        encoder = load_model("models/encoder_CMD_SPEED.keras")
    elif(feature == 'TORQUE'):
        autoencoder = load_model("models/autoencoder_TORQUE.keras")
        encoder = load_model("models/encoder_TORQUE.keras")

    kmeans = KMeans(n_clusters=2, random_state=0) # 2-Cluster-KMeans for peak vs. non-peak

    for sequence in range(8):
        for axis in [1,2,3,6]:
            signal = df_list[sequence][feature+'|'+str(axis)]
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

            plt.figure(figsize=(10, 4))
            plt.plot(target, label='Current-Signal')
            plt.plot(peaks_pos, target.iloc[peaks_pos], "rx", label='Peaks')
            plt.grid(True)
            plt.legend(fontsize=12)
            plt.tick_params(axis='both', labelsize=12)
            plt.savefig(f"{save_dir}/sequence{sequence}_axis{axis}.png", dpi=300, bbox_inches='tight')
            # plt.show()
            plt.close()
                    

    print("DBSCAN completed")
else:
    print("Skipping DBSCAN")