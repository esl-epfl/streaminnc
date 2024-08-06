import numpy as np
from config import Config

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.utils import shuffle
from sklearn.model_selection import LeaveOneGroupOut

from preprocessing import preprocessing_Dalia_aligned_preproc as pp


from self_attention_padding_study_models import build_model

import seaborn as sns

import pandas as pd

import time

import matplotlib.pyplot as plt

def get_nrmse(output, stream_output):
    error = output - stream_output
    nrmse = np.sqrt(np.mean(error**2)) / (np.max(output) - np.min(output))
    
    return nrmse


def clone_layer(layer):
    config = layer.get_config()
    weights = layer.get_weights()
    cloned_layer = type(layer).from_config(config)
    cloned_layer.build(layer.input_shape)
    cloned_layer.set_weights(weights)
    
    return cloned_layer

def clone_conv_block(conv_block, input_shape, batch_size = None):
    mInput = tf.keras.Input(shape = input_shape, batch_size = batch_size)
   
    m = mInput
    for i in range(1, len(conv_block.layers)):

        m = clone_layer(conv_block.layers[i])(m)
    
    return tf.keras.models.Model(inputs = mInput, 
                                 outputs = m)

def unroll_model(m, model):
    
    conv_outputs = []
    for i in range(1, len(model.layers)):
        m = clone_layer(model.layers[i])(m)
        
        current_layer_type = model.layers[i].__class__.__name__
        if current_layer_type == 'Conv1D':
            conv_outputs.append(m)
    return m, conv_outputs    

def get_extended_input(X, y, groups, activity):
    
    new_Xs = []
    new_ys = []
    new_groups = []
    new_activity = []
    
    
    for group in np.unique(groups):
        cur_X = X[groups == group]
        cur_y = y[groups == group]
        cur_groups = groups[groups == group]
        cur_activity = activity[groups == group]
        
        tmp = np.concatenate([cur_X[:-4, :256, :], 
                              cur_X[4:, -256:, :]], axis = 1)
        
        new_Xs.append(tmp)
        new_ys.append(cur_y[4:])
        new_groups.append(cur_groups[4:])
        new_activity.append(cur_activity[4:])
    
    new_Xs = np.concatenate(new_Xs, axis = 0)
    new_ys = np.concatenate(new_ys, axis = 0)
    new_groups = np.concatenate(new_groups, axis = 0)
    new_activity = np.concatenate(new_activity, axis = 0)
    
    return new_Xs, new_ys, new_groups, new_activity


tf.keras.utils.set_random_seed(0) 
tf.config.experimental.enable_op_determinism()

sns.set_theme()

cm = 1/2.54
figsize = (4 * cm, 4 * cm)

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

save_figures = False

input_size = 512

n_epochs = 200
batch_size = 256
n_ch = 1

test_subject_id = 5 #15

# Setup config
cf = Config(search_type = 'NAS', root = './data/')

# Load data
X, y, groups, activity = pp.preprocessing(cf.dataset, cf)
X = np.transpose(X, axes = (0, 2, 1))

X, y, groups, activity = get_extended_input(X, y, groups, activity)


X_train = X[groups != test_subject_id]
y_train = y[groups != test_subject_id]

X_test = X[groups == test_subject_id]
y_test = y[groups == test_subject_id]

X_validate = X_test
y_validate = y_test
activity_validate = activity[groups == test_subject_id]
        
# Build Model
model = build_model((input_size, n_ch),)
model.load_weights('./saved_models/signal_padding/model_weights/model_S' + str(int(test_subject_id)))


mInput = tf.keras.Input((input_size, 1))

m = mInput
m, conv_block_outputs1 = unroll_model(m, model.layers[1])
m, conv_block_outputs2 = unroll_model(m, model.layers[2])
m, conv_block_outputs3 = unroll_model(m, model.layers[3])

conv_outputs = conv_block_outputs1
conv_outputs.extend(conv_block_outputs2)
conv_outputs.extend(conv_block_outputs3)

submodel = tf.keras.models.Model(inputs = mInput, 
                                 outputs = [m, conv_block_outputs1])

with tf.device('/cpu:0'):
    y_pred = model.predict(X_validate)
    y_pred2 = submodel.predict(X_validate)

X_emb_1 = y_pred2[1][0]
X_emb_2 = y_pred2[1][8]


half = int(X_emb_2.shape[1]/2)

channel = 59
index = 205

plt.figure(figsize = figsize)
plt.plot( X_emb_2[index, 8:, channel][half:], linewidth = linewidth)
plt.plot(X_emb_2[index + 1, :-8, channel][half:], linewidth = linewidth)

plt.xlabel('Activation sample')
plt.ylabel('Activation value')

if save_figures:
    plt.savefig('./results/figures/zero_padding_effects/signal_padding_activations.svg', 
                bbox_inches = 'tight')
    
print(get_nrmse(X_emb_2[index, 8:, channel][half:], 
                X_emb_2[index + 1, :-8, channel][half:]) * 100)

