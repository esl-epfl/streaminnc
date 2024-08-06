from pathlib import Path
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt


import scipy
from scipy import signal

import time
import pickle
from tqdm import tqdm

import os


def get_nrmse(output, stream_output):
    error = output - stream_output
    nrmse = np.sqrt(np.mean(error**2)) / (np.max(output) - np.min(output))
    
    return nrmse


def plot_fft(y, fs = 32.0, linewidth = None, label = None):
    N = y.size
    
    # sample spacing
    T = 1/fs
    x = np.linspace(0.0, N*T, N)
    yf = scipy.fftpack.fft(y)
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2) * 60
    
    plt.plot(xf, 2.0/N * np.abs(yf[:N//2]), linewidth = linewidth, label = label)

    
def get_all_patient_filenames(patient_path):
    return [f for f in os.listdir(patient_path) if os.path.isfile(os.path.join(patient_path, f))]

batch_size = 1024

fs = 32.0

# load a time-serie
data_path = Path("./gtcs-detection-model-export/seizure-subset/")

center = "dianalund"
patient = 19
# recording = "1597068172_A0249F"

recordings = get_all_patient_filenames(data_path/center/f"patient_{patient}")

print('Processing ' + center + ' Patient ' + str(patient))


nrmses_ch0s = []
nrmses_ch1s = []

for recording in recordings:
    print('\t Recording ' + recording)

    
    result_file_name = './results/' + center + '/patient_' + str(patient)\
        + '/' 
        
    result_file_name += recording + '.pickle'
        
    with open(result_file_name, 'rb') as handle:
        results = pickle.load(handle)
        
    nrmse_ch0 = get_nrmse(results['y_pred'][:, 0], 
                          results['y_pred_shift'][:, 0])
    nrmse_ch1 = get_nrmse(results['y_pred'][:, 1], 
                          results['y_pred_shift'][:, 1]) 
    nrmses_ch0s.append(nrmse_ch0.mean())
    nrmses_ch1s.append(nrmse_ch1.mean())
    
nrmses_ch0s = np.array(nrmses_ch0s)
nrmses_ch1s = np.array(nrmses_ch1s)

print("NRMSE CH0 ", nrmses_ch0s.mean() * 100)
print("NRMSE CH1 ", nrmses_ch1s.mean() * 100)

nrmses_ch0s_approx = []
nrmses_ch1s_approx = []

for recording in recordings:
    print('\t Recording ' + recording)

    result_file_name = './results/' + center + '/patient_' + str(patient)\
        + '/' 
        
    result_file_name += recording + '_approximate.pickle'
        
    with open(result_file_name, 'rb') as handle:
        results_approx = pickle.load(handle)
        
    nrmse_ch0 = get_nrmse(results_approx['y_pred'][:, 0], 
                          results_approx['y_pred_shift'][:, 0])
    nrmse_ch1 = get_nrmse(results_approx['y_pred'][:, 1], 
                          results_approx['y_pred_shift'][:, 1]) 
    nrmses_ch0s_approx.append(nrmse_ch0.mean())
    nrmses_ch1s_approx.append(nrmse_ch1.mean())
    
nrmses_ch0s_approx = np.array(nrmses_ch0s_approx)
nrmses_ch1s_approx = np.array(nrmses_ch1s_approx)

print("NRMSE CH0 ", nrmses_ch0s_approx.mean() * 100)
print("NRMSE CH1 ", nrmses_ch1s_approx.mean() * 100)

plt.figure()
plt.plot(results['y_pred'][:, 0], label = 'Original Inference')

plt.plot(results['y_pred_shift'][:, 0], 
         label = 'Streaming Inference')
plt.plot(results_approx['y_pred_shift'][:, 0], 
         label = 'Approx. Streaming Inference')
plt.legend()

