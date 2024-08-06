import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import scipy
        
sns.set_theme()

cm = 1/2.54
figsize = (3 * cm, 3 * cm)

fontsize = 6

markersize = 2.5
linewidth = 1.5

plt.rc('font', size = fontsize)          # controls default text sizes
plt.rc('axes', titlesize = fontsize)     # fontsize of the axes title
plt.rc('axes', labelsize = fontsize)    # fontsize of the x and y labels
plt.rc('xtick', labelsize = fontsize)    # fontsize of the tick labels
plt.rc('ytick', labelsize = fontsize)    # fontsize of the tick labels
plt.rc('legend', fontsize = fontsize)    # legend fontsize
plt.rc('figure', titlesize = fontsize)  # fontsize of the figure title

plt.rcParams.update({"font.family" : "Times New Roman"})

save_figures = True

n_freq_terms = 1
window_length_sec = 8
f = 1
pool_size = 8


all_fs = np.array([64, 128, 256, 512, 1024,])
max_error_fs = np.zeros((len(all_fs)))
mean_error_fs = np.zeros((len(all_fs)))
error_bounds = np.zeros((len(all_fs)))

for i, fs in enumerate(all_fs):
   
    window_length_samples = int(window_length_sec * fs)
    
    max_cur_error = []
    mean_cur_error = []
    for window_step_samples in range(1, pool_size + 1):
        t = np.linspace(0, 2 * window_length_sec, 2 * window_length_samples)
        
        y = 0
        
        for j in range(1, n_freq_terms + 1):
            y += np.cos(2 * np.pi * j * f * t)
            
    
        X1 = y[:window_length_samples]
        X2 = y[window_step_samples : window_length_samples + window_step_samples]
            
        maxPooling = tf.keras.layers.MaxPool1D(pool_size = pool_size)
        
        X1_mp = maxPooling(X1[None, :, None]).numpy().flatten()
        X2_mp = maxPooling(X2[None, :, None]).numpy().flatten()
        
        X1_mp_common = X1_mp[window_step_samples // pool_size: ]
        X2_mp_common = X2_mp[ : - window_step_samples // pool_size]
        
        n = np.min([X1_mp_common.size, X2_mp_common.size])
        X1_mp_common = X1_mp_common[:n]
        X2_mp_common = X2_mp_common[:n]
        
        # error = np.sqrt(np.sum((X1_mp_common - X2_mp_common)**2)) / np.sqrt(np.sum((X1_mp_common)**2))
        max_error = np.max(np.abs(X1_mp_common - X2_mp_common)) 
        mean_error = np.mean(np.abs(X1_mp_common - X2_mp_common)) 
        
        max_cur_error.append(max_error)
        mean_cur_error.append(mean_error)
        
    max_error_fs[i] = np.max(max_cur_error)
    mean_error_fs[i] = np.max(mean_cur_error)
    
    error_bounds[i] = (pool_size - 1) * 2 * np.pi \
        * (n_freq_terms) * (n_freq_terms * f) / (fs )
    
plt.figure(figsize = figsize)
plt.plot(np.log2(all_fs), mean_error_fs, '-o', label = 'Max Mean Error',
         markersize = markersize, linewidth = linewidth)
plt.plot(np.log2(all_fs), max_error_fs, '-o', label = 'Empirical Max Error',
         markersize = markersize, linewidth = linewidth)
plt.plot(np.log2(all_fs), error_bounds, '-o', label = 'Upper Bound',
         markersize = markersize, linewidth = linewidth)

plt.xlabel('log2 Sampling Frequency')
plt.ylabel('Relative Error')

if save_figures:
    plt.savefig('./results/figures/pooling_error_bounds/error_vs_fs_one_osc.svg',
                bbox_inches = 'tight')

n_freq_terms = 5
window_length_sec = 8
f = 1
pool_size = 8


all_fs = np.array([64, 128, 256, 512, 1024,])
max_error_fs = np.zeros((len(all_fs)))
mean_error_fs = np.zeros((len(all_fs)))
error_bounds = np.zeros((len(all_fs)))

for i, fs in enumerate(all_fs):
   
    window_length_samples = int(window_length_sec * fs)
    
    max_cur_error = []
    mean_cur_error = []
    for window_step_samples in range(1, pool_size + 1):
        t = np.linspace(0, 2 * window_length_sec, 2 * window_length_samples)
        
        y = 0
        
        for j in range(1, n_freq_terms + 1):
            y += np.cos(2 * np.pi * j * f * t)
            
    
        X1 = y[:window_length_samples]
        X2 = y[window_step_samples : window_length_samples + window_step_samples]
            
        maxPooling = tf.keras.layers.MaxPool1D(pool_size = pool_size)
        
        X1_mp = maxPooling(X1[None, :, None]).numpy().flatten()
        X2_mp = maxPooling(X2[None, :, None]).numpy().flatten()
        
        X1_mp_common = X1_mp[window_step_samples // pool_size: ]
        X2_mp_common = X2_mp[ : - window_step_samples // pool_size]
        
        n = np.min([X1_mp_common.size, X2_mp_common.size])
        X1_mp_common = X1_mp_common[:n]
        X2_mp_common = X2_mp_common[:n]
        
        # error = np.sqrt(np.sum((X1_mp_common - X2_mp_common)**2)) / np.sqrt(np.sum((X1_mp_common)**2))
        max_error = np.max(np.abs(X1_mp_common - X2_mp_common)) / np.max(np.abs(y))
        mean_error = np.mean(np.abs(X1_mp_common - X2_mp_common)) / np.max(np.abs(y))
        
        max_cur_error.append(max_error)
        mean_cur_error.append(mean_error)
        
    max_error_fs[i] = np.max(max_cur_error)
    mean_error_fs[i] = np.max(mean_cur_error)
    
    error_bounds[i] = (pool_size - 1) * 2 * np.pi \
        * (n_freq_terms) * (n_freq_terms * f) / (fs )
    error_bounds[i] = error_bounds[i] / n_freq_terms
    
plt.figure(figsize = figsize)
plt.plot(np.log2(all_fs), mean_error_fs, '-o', label = 'Mean Error',
         markersize = markersize, linewidth = linewidth)
plt.plot(np.log2(all_fs), max_error_fs, '-o', label = 'Empirical Max Error',
         markersize = markersize, linewidth = linewidth)
plt.plot(np.log2(all_fs), error_bounds, '-o', label = 'Upper Bound',
         markersize = markersize, linewidth = linewidth)
# plt.legend()

plt.xlabel('log2 Sampling Frequency')
plt.ylabel('Relative Error')

if save_figures:
    plt.savefig('./results/figures/pooling_error_bounds/error_vs_fs_five_osc.svg',
                bbox_inches = 'tight')


n_freq_terms = 1
window_length_sec = 8
f = 1


fs = 256
all_pool_sizes = np.array([2, 4, 8, 16])
max_error_fs = np.zeros((len(all_pool_sizes)))
mean_error_fs = np.zeros((len(all_pool_sizes)))
error_bounds = np.zeros((len(all_pool_sizes)))

for i, pool_size in enumerate(all_pool_sizes):
   
    window_length_samples = int(window_length_sec * fs)
    
    max_cur_error = []
    mean_cur_error = []
    for window_step_samples in range(1, pool_size + 1):
        t = np.linspace(0, 2 * window_length_sec, 2 * window_length_samples)
        
        y = 0
        
        for j in range(1, n_freq_terms + 1):
            y += np.cos(2 * np.pi * j * f * t)
            
    
        X1 = y[:window_length_samples]
        X2 = y[window_step_samples : window_length_samples + window_step_samples]
            
        maxPooling = tf.keras.layers.MaxPool1D(pool_size = int(pool_size))
        
        X1_mp = maxPooling(X1[None, :, None]).numpy().flatten()
        X2_mp = maxPooling(X2[None, :, None]).numpy().flatten()
        
        X1_mp_common = X1_mp[window_step_samples // pool_size: ]
        X2_mp_common = X2_mp[ : - window_step_samples // pool_size]
        
        n = np.min([X1_mp_common.size, X2_mp_common.size])
        X1_mp_common = X1_mp_common[:n]
        X2_mp_common = X2_mp_common[:n]
        
        # error = np.sqrt(np.sum((X1_mp_common - X2_mp_common)**2)) / np.sqrt(np.sum((X1_mp_common)**2))
        max_error = np.max(np.abs(X1_mp_common - X2_mp_common)) 
        mean_error = np.mean(np.abs(X1_mp_common - X2_mp_common)) 
        
        max_cur_error.append(max_error)
        mean_cur_error.append(mean_error)
        
    max_error_fs[i] = np.max(max_cur_error)
    mean_error_fs[i] = np.max(mean_cur_error)
    
    error_bounds[i] = (pool_size - 1) * 2 * np.pi \
        * (n_freq_terms) * (n_freq_terms * f) / (fs )
    
plt.figure(figsize = figsize)
plt.plot(np.log2(all_pool_sizes), mean_error_fs, '-o', label = 'Mean Error',
         markersize = markersize, linewidth = linewidth)
plt.plot(np.log2(all_pool_sizes), max_error_fs, '-o', 
         label = 'Empirical Max Error',
                  markersize = markersize, linewidth = linewidth)
plt.plot(np.log2(all_pool_sizes), error_bounds, '-o', label = 'Upper Bound',
         markersize = markersize, linewidth = linewidth)
# plt.legend()

plt.xlabel('log2 Pooling Window Length')
plt.ylabel('Relative Error')

if save_figures:
    plt.savefig('./results/figures/pooling_error_bounds/error_vs_poolsize_one_osc.svg',
                bbox_inches = 'tight')

n_freq_terms = 5
window_length_sec = 8
f = 1


fs = 256
all_pool_sizes = np.array([2, 4, 8, 16])
max_error_fs = np.zeros((len(all_pool_sizes)))
mean_error_fs = np.zeros((len(all_pool_sizes)))
error_bounds = np.zeros((len(all_pool_sizes)))

for i, pool_size in enumerate(all_pool_sizes):
   
    window_length_samples = int(window_length_sec * fs)
    
    max_cur_error = []
    mean_cur_error = []
    for window_step_samples in range(1, pool_size + 1):
        t = np.linspace(0, 2 * window_length_sec, 2 * window_length_samples)
        
        y = 0
        
        for j in range(1, n_freq_terms + 1):
            y += np.cos(2 * np.pi * j * f * t)
            
    
        X1 = y[:window_length_samples]
        X2 = y[window_step_samples : window_length_samples + window_step_samples]
            
        maxPooling = tf.keras.layers.MaxPool1D(pool_size = int(pool_size))
        
        X1_mp = maxPooling(X1[None, :, None]).numpy().flatten()
        X2_mp = maxPooling(X2[None, :, None]).numpy().flatten()
        
        X1_mp_common = X1_mp[window_step_samples // pool_size: ]
        X2_mp_common = X2_mp[ : - window_step_samples // pool_size]
        
        n = np.min([X1_mp_common.size, X2_mp_common.size])
        X1_mp_common = X1_mp_common[:n]
        X2_mp_common = X2_mp_common[:n]
        
        # error = np.sqrt(np.sum((X1_mp_common - X2_mp_common)**2)) / np.sqrt(np.sum((X1_mp_common)**2))
        max_error = np.max(np.abs(X1_mp_common - X2_mp_common)) / np.max(np.abs(y))
        mean_error = np.mean(np.abs(X1_mp_common - X2_mp_common)) / np.max(np.abs(y))
        
        max_cur_error.append(max_error)
        mean_cur_error.append(mean_error)
        
    max_error_fs[i] = np.max(max_cur_error)
    mean_error_fs[i] = np.max(mean_cur_error)
    
    error_bounds[i] = (pool_size - 1) * 2 * np.pi \
        * (n_freq_terms) * (n_freq_terms * f) / (fs )
    error_bounds[i] = error_bounds[i] / n_freq_terms
    
plt.figure(figsize = figsize)
plt.plot(np.log2(all_pool_sizes), mean_error_fs, '-o', label = 'Mean Error',
         markersize = markersize, linewidth = linewidth)
plt.plot(np.log2(all_pool_sizes), max_error_fs, '-o', 
         label = 'Empirical Max Error',
                  markersize = markersize, linewidth = linewidth)
plt.plot(np.log2(all_pool_sizes), error_bounds, '-o', label = 'Upper Bound',
         markersize = markersize, linewidth = linewidth)
# plt.legend()

plt.xlabel('log2 Pooling Window Length')
plt.ylabel('Relative Error')

if save_figures:
    plt.savefig('./results/figures/pooling_error_bounds/error_vs_poolsize_five_osc.svg',
                bbox_inches = 'tight')
