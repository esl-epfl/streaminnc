import numpy as np
from config import Config

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.utils import shuffle
from sklearn.model_selection import LeaveOneGroupOut

from preprocessing import preprocessing_Dalia_aligned_preproc as pp

import tensorflow_probability as tfp
tfd = tfp.distributions
from self_attention_ppg_only_models import build_attention_model
from self_attention_padding_study_models import build_model


import pandas as pd

import time

import pickle

import matplotlib.pyplot as plt
import scipy
from scipy import fftpack
from tqdm import tqdm

import os

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


n_epochs = 200
batch_size = 256
n_ch = 1

for test_subject_id in range(1, 16):
    
    print("Processing ", test_subject_id, "....")
    # Setup config
    cf = Config(search_type = 'NAS', root = './data/')
    
    # Load data
    X, y, groups, activity = pp.preprocessing(cf.dataset, cf)
    
    X_exp = np.transpose(X, axes = (0, 2, 1))
    X_exp, y_exp, groups_exp, activity_exp = get_extended_input(X_exp, y, groups, activity)

    X_exp_test = X_exp[groups_exp == test_subject_id]
    y_exp_test = y_exp[groups_exp == test_subject_id]

    X_exp_validate = X_exp_test
    y_exp_validate = y_exp_test

    
    X_train = X[groups != test_subject_id]
    y_train = y[groups != test_subject_id]
    
    X_test = X[groups == test_subject_id]
    y_test = y[groups == test_subject_id]
    
    X_validate = X_test
    y_validate = y_test
    activity_validate = activity[groups == test_subject_id]
    
    # Build Model
    model = build_attention_model((cf.input_shape, n_ch))
    
    
    shift_model = build_model((512, n_ch),)
    shift_model.load_weights('./saved_models/signal_padding/model_weights/model_S' + str(int(test_subject_id)))
    
    model.set_weights(shift_model.get_weights())
    
    X_validate = X_validate[:, :1, :]
    
    X_validate = np.transpose(X_validate, axes = (0, 2, 1))

    
    conv_block1 = tf.keras.models.Model(inputs = model.layers[1].inputs,
                                          outputs = model.layers[1].outputs)
    conv_block2 = tf.keras.models.Model(inputs = model.layers[2].inputs,
                                          outputs = model.layers[2].outputs)
    conv_block3 = tf.keras.models.Model(inputs = model.layers[3].inputs,
                                          outputs = model.layers[3].outputs)
    
    attention = tf.keras.models.Model(inputs = model.layers[4].input,
                                          outputs = model.layers[4].output)
    
    hr_inference_sb = tf.keras.models.Model(inputs = model.layers[5].input,
                                          outputs = model.outputs)
    
    mInput = tf.keras.Input((256, 1))
    m = conv_block1(mInput)
    m = conv_block2(m)
    m = conv_block3(m)
    
    m = attention(m)
    m_emb_out = m
    
    m = hr_inference_sb(m)
    
    conv1 = model.layers[1].layers[1]
    conv2 = model.layers[1].layers[2]
    conv3 = model.layers[1].layers[3]
    
    conv21 = model.layers[2].layers[1]
    conv22 = model.layers[2].layers[2]
    conv23 = model.layers[2].layers[3]
    
    conv31 = model.layers[3].layers[1]
    conv32 = model.layers[3].layers[2]
    conv33 = model.layers[3].layers[3]
    
    conv_block2 = tf.keras.models.Model(inputs = model.layers[2].inputs,
                                          outputs = model.layers[2].outputs)
    conv_block3 = tf.keras.models.Model(inputs = model.layers[3].inputs,
                                          outputs = model.layers[3].outputs)
    
    hr_inference_sb = tf.keras.models.Model(inputs = model.layers[4].input,
                                          outputs = model.outputs)
    
    mInput = tf.keras.Input((16, 64))
    m = mInput
    m = hr_inference_sb(m)
    hr_inference = tf.keras.models.Model(inputs = mInput, outputs = m)
    
    sample_shift = 64
    
    samples_indexes = np.arange(1, X_validate.shape[0])
    
    with tf.device("/:cpu0"):
        y_pred = shift_model.predict(X_exp_validate)
        
        prev_act_1 = conv1(X_validate[:1, ...])
        prev_act_2 = conv2(prev_act_1)
        prev_act_3 = conv3(prev_act_2)
        
        prev_act_4_pool = tf.keras.layers.AveragePooling1D(pool_size = 4)(prev_act_3)
        prev_act_4 = conv21(prev_act_4_pool)
        prev_act_5 = conv22(prev_act_4)
        prev_act_6 = conv23(prev_act_5)
        
        prev_act_7_pool = tf.keras.layers.AveragePooling1D(pool_size = 2)(prev_act_6)
        prev_act_7 = conv31(prev_act_7_pool)
        prev_act_8 = conv32(prev_act_7)
        prev_act_9 = conv33(prev_act_8)
        prev_act_10 = tf.keras.layers.AveragePooling1D(pool_size = 2)(prev_act_9)
        
        y_pred_shift = np.zeros(samples_indexes.size + 1)
        y_pred_shift[0] = y_pred[0]
        
        for i in tqdm(samples_indexes):
            curX = X_validate[i:i+1, ...]
            
            output_tmp1 = conv1(curX)
            output1 = tf.concat([prev_act_1[:, sample_shift:, :], 
                                  output_tmp1[:, -sample_shift:, :]], axis = 1)
    
            
            output_tmp2 = conv2(output1)
            output2 = tf.concat([prev_act_2[:, sample_shift:, :], 
                                  output_tmp2[:, -sample_shift:, :]], axis = 1)
            
            
            output_tmp3 = conv3(output2)
            output3 = tf.concat([prev_act_3[:, sample_shift:, :], 
                                  output_tmp3[:, -sample_shift:, :]], axis = 1)
    
            
            # Conv Block 2
            
            output_tmp4_pool = tf.keras.layers.AveragePooling1D(pool_size = 4)(output3)
            output4_pool = tf.concat([prev_act_4_pool[:, int(sample_shift / 4):, :], 
                                  output_tmp4_pool[:, -int(sample_shift / 4):, :]], axis = 1)
    
            output_tmp4 = conv21(output4_pool)
            output4 = tf.concat([prev_act_4[:, int(sample_shift / 4):, :], 
                                  output_tmp4[:, -int(sample_shift / 4):, :]], axis = 1)
            
            output_tmp5 = conv22(output4)
            output5 = tf.concat([prev_act_5[:, int(sample_shift / 4):, :], 
                                  output_tmp5[:, -int(sample_shift / 4):, :]], axis = 1)
            
            output_tmp6 = conv23(output5)
            output6 = tf.concat([prev_act_6[:, int(sample_shift / 4):, :], 
                                  output_tmp6[:, -int(sample_shift / 4):, :]], axis = 1)
            
            # Conv Block 3
            
            output_tmp7_pool = tf.keras.layers.AveragePooling1D(pool_size = 2)(output6)
            output7_pool = tf.concat([prev_act_7_pool[:, int((sample_shift / 4) / 2):, :], 
                                  output_tmp7_pool[:, -int((sample_shift / 4) / 2):, :]], axis = 1)
           
            output_tmp7 = conv31(output7_pool)
            output7 = tf.concat([prev_act_7[:, int((sample_shift / 4) / 2):, :], 
                                  output_tmp7[:, -int((sample_shift / 4) / 2):, :]], axis = 1)
            
            output_tmp8 = conv32(output7)
            output8 = tf.concat([prev_act_8[:, int((sample_shift / 4) / 2):, :], 
                                  output_tmp8[:, -int((sample_shift / 4) / 2):, :]], axis = 1)
            
            output_tmp9 = conv33(output8)
            output9 = tf.concat([prev_act_9[:, int((sample_shift / 4) / 2):, :], 
                                  output_tmp9[:, -int((sample_shift / 4) / 2):, :]], axis = 1)
            
            output10_tmp = tf.keras.layers.AveragePooling1D(pool_size = 2)(output9)
            output10 = tf.concat([prev_act_10[:, int((sample_shift / 4) / 2 / 2):, :], 
                                  output10_tmp[:, -int((sample_shift / 4) / 2 / 2):, :]], axis = 1)
            
            prev_act_1 = output1
            prev_act_2 = output2
            prev_act_3 = output3
            
            prev_act_4_pool = output4_pool
            prev_act_4 = output4
            prev_act_5 = output5
            prev_act_6 = output6
            
            prev_act_7_pool = output7_pool
            prev_act_7 = output7
            prev_act_8 = output8
            prev_act_9 = output9
            prev_act_10 = output10
            
            y_pred_shift[i] = hr_inference(output10).numpy()[0]
    
    
    results = {'y_test' : y_test.flatten(),
               'y_pred' : y_pred.flatten(),
               'y_pred_shift' : y_pred_shift.flatten()}
    
    output_file_name = './results/signal_padding/S' + str(test_subject_id) + '.pickle'
            
        
    with open(output_file_name, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
