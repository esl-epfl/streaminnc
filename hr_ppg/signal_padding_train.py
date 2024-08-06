import numpy as np
from config import Config

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.utils import shuffle
from sklearn.model_selection import LeaveOneGroupOut

from preprocessing import preprocessing_Dalia_aligned_preproc as pp


from self_attention_padding_study_models import build_model

import pandas as pd

import time

def get_session(gpu_fraction=0.333):
    gpu_options = tf.compat.v1.GPUOptions(
            per_process_gpu_memory_fraction=gpu_fraction,
            allow_growth=True)
    return tf.compat.v1.Session(
            config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
tf.compat.v1.keras.backend.set_session(get_session())

tf.keras.utils.set_random_seed(0) 
tf.config.experimental.enable_op_determinism()

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
        

n_epochs = 100
batch_size = 256
n_ch = 1

input_size = 512

test_subject_id = 1

for test_subject_id in range(1, 16):
    
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
    
    
    adam = Adam(learning_rate = 0.0005, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08)
    
    model.compile(loss = 'mae', 
                  optimizer = adam, metrics='mae')
    
    X_train, y_train = shuffle(X_train, y_train)
    
    # Training
    hist = model.fit(
        x = X_train,
        y = y_train,
        epochs = n_epochs, 
        batch_size = batch_size,
        validation_data = (X_validate, 
                           y_validate), 
        
        verbose = 1)
    
    model.save_weights('./saved_models/signal_padding/model_weights/model_S' + str(int(test_subject_id)))
