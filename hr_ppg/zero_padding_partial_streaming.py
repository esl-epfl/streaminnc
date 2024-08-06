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

import pandas as pd

import time

import pickle

import matplotlib.pyplot as plt
import scipy
from scipy import fftpack
from tqdm import tqdm

n_epochs = 200
batch_size = 256
n_ch = 1

for test_subject_id in range(1, 16):
    # Setup config
    cf = Config(search_type = 'NAS', root = './data/')

    # Load data
    X, y, groups, activity = pp.preprocessing(cf.dataset, cf)


    X_train = X[groups != test_subject_id]
    y_train = y[groups != test_subject_id]

    X_test = X[groups == test_subject_id]
    y_test = y[groups == test_subject_id]

    X_validate = X_test
    y_validate = y_test
    activity_validate = activity[groups == test_subject_id]

    # Build Model
    model = build_attention_model((cf.input_shape, n_ch))



    model.load_weights('./saved_models/adaptive_w_attention/model_weights/model_S' + str(int(test_subject_id)) + '.h5')

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

    mInput = tf.keras.Input((64, 32))
    m = conv_block2(mInput)
    m = conv_block3(m)
    m = hr_inference_sb(m)
    hr_inference = tf.keras.models.Model(inputs = mInput, outputs = m)

    sample_shift = 64

    samples_indexes = np.arange(1, X_validate.shape[0])
    # samples_indexes = np.arange(1, 250)

    with tf.device("/:cpu0"):
        y_pred = model.predict(X_validate)
        
        prev_act_1 = conv1(X_validate[:1, ...])
        prev_act_2 = conv2(prev_act_1)
        prev_act_3 = conv3(prev_act_2)
        
        prev_act_4_pool = tf.keras.layers.AveragePooling1D(pool_size = 4)(prev_act_3)

        y_pred_shift = np.zeros(samples_indexes.size + 1)
        y_pred_shift[0] = y_pred[0]
        
        for i in tqdm(samples_indexes, ascii = True):
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
    
            prev_act_1 = output1
            prev_act_2 = output2
            prev_act_3 = output3
            
            prev_act_4_pool = output4_pool
            
            y_pred_shift[i] = hr_inference(prev_act_4_pool).numpy()[0]

    results = {'y_test' : y_test.flatten(),
                'y_pred' : y_pred.flatten(),
                'y_pred_shift' : y_pred_shift.flatten()}
            
    output_file_name = './results/zero_padding_partial_streaming/S' + str(test_subject_id) + '.pickle'
                
            
    with open(output_file_name, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
        