
import os
import librosa
import librosa.display
import IPython.display as ipd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # or another backend like 'agg'
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
warnings.filterwarnings("ignore", category=RuntimeWarning)

scale_file = "D:/oceanvue.fyp/DeepShip Dataset/Cargo/1/1.wav"
ipd.Audio(scale_file)

# load audio files with librosa
scale, sr = librosa.load(scale_file)

#parameters
FRAME_SIZE = 2048
HOP_SIZE = 512

S_scale = librosa.stft(scale, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)

Y_scale = np.abs(S_scale) ** 2

import matplotlib.pyplot as plt
import numpy as np

def plot_spectrogram(Y, sr, hop_length, y_axis="linear"):
    plt.figure(figsize=(25, 10))
    plt.imshow(np.abs(Y), aspect="auto", origin="lower", cmap="viridis", extent=[0, Y.shape[1] * hop_length / sr, 0, sr / 2])
    plt.colorbar(format="%+2.f")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Spectrogram")
    plt.show()

plot_spectrogram(Y_scale, sr, HOP_SIZE)
