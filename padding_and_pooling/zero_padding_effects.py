import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

def calculate_padding_effect_nonones_count(y):
    dist_vs_layer = np.zeros((len(y)))
    for i in range(len(y)):
        d = np.mean(y[i] < 1) * 100
        dist_vs_layer[i] = d
    return dist_vs_layer


def get_rl_limit(I, O, kl, g):
    I = 256
    O = 64

    ratio = []

    k0 = kl[0]
    ratio.append((I - O) / k0)

    for l in range(1, 9):
        s = k0
        for i in range(1, l + 1):
            s += kl[i] * np.prod(g[:i])
        ratio.append((I - O) / s)    
        
    return ratio

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


###################
### Model HR model
###################

input_size = 256
kernel_size = 10
padding = 'causal'

mInput = tf.keras.Input(shape = (input_size, 1))

outputs = []

m = tf.keras.layers.Conv1D(filters = 1, 
                           kernel_size = kernel_size,
                           padding = padding)(mInput)
outputs.append(m)
for i in range(2):
    m = tf.keras.layers.Conv1D(filters = 1, 
                               kernel_size = kernel_size,
                               padding = padding)(m)
    outputs.append(m)

m = tf.keras.layers.AveragePooling1D(pool_size = 4)(m)

for i in range(3):
    m = tf.keras.layers.Conv1D(filters = 1, 
                               kernel_size = kernel_size,
                               padding = padding)(m)
    outputs.append(m)
    
m = tf.keras.layers.AveragePooling1D(pool_size = 2)(m)

for i in range(3):
    m = tf.keras.layers.Conv1D(filters = 1, 
                               kernel_size = kernel_size,
                               padding = padding)(m)
    outputs.append(m)

model_hr = tf.keras.models.Model(inputs = mInput, 
                              outputs = outputs)

kernel = np.ones([kernel_size, 1, 1])
kernel = kernel / kernel.sum()
bias = np.zeros((1,))

weights = [kernel, bias]

for i in range(1, len(model_hr.layers)):
    if model_hr.layers[i].__class__.__name__ == 'Conv1D':
        model_hr.layers[i].set_weights(weights)

x = np.ones((1, input_size, 1))

with tf.device('/cpu:0'):
    y_hr = model_hr.predict(x)
    
conv_names = ['conv_' + str(i) for i in range(1, 10)]
    
plt.figure(figsize = (figsize))
for i in range(len(y_hr)):
    t = np.linspace(0, 1, y_hr[i].flatten().size)
    plt.plot(t, y_hr[i].flatten(), label = conv_names[i], 
             linewidth = linewidth, markersize = markersize)
plt.legend(labelspacing = 0.001)
plt.xlabel('Output Sample (% of output length)')
plt.ylabel('Convolution Output')

if save_figures:
    plt.savefig('./results/figures/zero_padding_effects/activations_effect_zero_padding.svg')

dist_vs_layer_hr_nones = calculate_padding_effect_nonones_count(y_hr)

###################
### Model ACC Epilepsy model
###################

epilepsy_kernel_size = 3

mInput = tf.keras.Input(shape = (960, 1))

epilepsy_model_outputs = []
m = mInput
for i in range(6):
    m = tf.keras.layers.Conv1D(filters = 1, kernel_size = epilepsy_kernel_size, 
                           padding = 'same', activation = None)(m)
    epilepsy_model_outputs.append(m)

epilepsy_model = tf.keras.models.Model(inputs = mInput,
                                       outputs = epilepsy_model_outputs)

kernel = np.ones([epilepsy_kernel_size, 1, 1])
kernel = kernel / kernel.sum()
bias = np.zeros((1,))

weights = [kernel, bias]

for i in range(1, len(epilepsy_model.layers)):
    if epilepsy_model.layers[i].__class__.__name__ == 'Conv1D':
        epilepsy_model.layers[i].set_weights(weights)

x = np.ones((1, 960, 1))
with tf.device('/cpu:0'):
    y_pred_epilespy = epilepsy_model.predict(x)
    
dist_vs_layer_epilepsy_nones = calculate_padding_effect_nonones_count(y_pred_epilespy)

###################
### Model EEG Epilepsy model
###################

epilepsy_eeg_kernel_size = 3

mInput = tf.keras.Input(shape = (1024, 1))

epilepsy_eeg_model_outputs = []
m = mInput
for i in range(3):
    m = tf.keras.layers.Conv1D(filters = 1, kernel_size = epilepsy_eeg_kernel_size, 
                           padding = 'same', activation = None)(m)
    epilepsy_eeg_model_outputs.append(m)
    m = tf.keras.layers.MaxPool1D(pool_size = 4)(m)

epilepsy_eeg_model = tf.keras.models.Model(inputs = mInput,
                                       outputs = epilepsy_eeg_model_outputs)

kernel = np.ones([epilepsy_eeg_kernel_size, 1, 1])
kernel = kernel / kernel.sum()
bias = np.zeros((1,))

weights = [kernel, bias]

for i in range(1, len(epilepsy_eeg_model.layers)):
    if epilepsy_eeg_model.layers[i].__class__.__name__ == 'Conv1D':
        epilepsy_eeg_model.layers[i].set_weights(weights)

x = np.ones((1, 1024, 1))
with tf.device('/cpu:0'):
    y_pred_epilespy_eeg = epilepsy_eeg_model.predict(x)
    
dist_vs_layer_epilepsy_eeg_nones = calculate_padding_effect_nonones_count(y_pred_epilespy_eeg)

plt.figure(figsize = (figsize))
plt.plot(dist_vs_layer_hr_nones, '-o', label = 'HR Convs', 
         linewidth = linewidth, markersize = markersize)
plt.plot(dist_vs_layer_epilepsy_eeg_nones, '-o', label = 'Epilepsy EEG Convs', 
         linewidth = linewidth, markersize = markersize)
plt.plot(dist_vs_layer_epilepsy_nones, '-o', label = 'Epilepsy ACC Convs', 
         linewidth = linewidth, markersize = markersize)
plt.grid()
plt.legend()

plt.xlabel('Conv Layer')
plt.ylabel('Percentage of output samples < 1')

if save_figures:
    plt.savefig('./results/figures/zero_padding_effects/percent_output_samples_notone.svg')
