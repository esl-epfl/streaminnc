from keras_model import define_model


from pathlib import Path
import pandas as pd

import numpy as np

import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.python.framework.convert_to_constants import  convert_variables_to_constants_v2_as_graph

import scipy
from scipy import signal

import time
import pickle
from tqdm import tqdm

import os


def get_flops(model, batch_size = None):
    
    if batch_size is None:
        b = 1
    else:
        b = batch_size
    
    concrete = tf.function(lambda inputs: model(inputs))
    concrete_func = concrete.get_concrete_function(
        [tf.TensorSpec([b, *inputs.shape[1:]]) for inputs in model.inputs])
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name='')
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd="op", options=opts)
        return flops.total_float_ops

def expand_model_outputs(model, output_layers = 'Conv1D'):
    mInput = model.inputs
    outputs = []
    outputs.append(model.outputs)
    
    for i in range(len(model.layers)):
        current_layer_type = model.layers[i].__class__.__name__
        if current_layer_type == output_layers:
            output = model.layers[i](model.layers[i - 1].output)
            outputs.append(output)
    new_model = tf.keras.models.Model(inputs = mInput, 
                                      outputs = outputs)
    
    return new_model

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
for recording in recordings:
    print('\t Recording ' + recording)
    #----------------------------------------------------------------------------------------------------------------------#
    # load gtcs info
    gtcs_df = pd.read_csv(data_path / "gtcs_subset.csv")
    gtcs_df = gtcs_df[(gtcs_df["center"] == center) & (gtcs_df["patient"] == patient) & (gtcs_df["E4_recording"] == recording)]
    
    # ensure the used columns are in the correct format
    gtcs_df['focal_onset'] = pd.to_datetime(gtcs_df['focal_onset'])
    gtcs_df['clonic_end'] = pd.to_datetime(gtcs_df['clonic_end'])
    
    # load recording
    recording_data = pd.read_csv(data_path / center / f"patient_{patient}" / recording,)
    
    
    keras_model = define_model()
    
    window_length = 960
    window_stride = 160 
    X_input = []
    for i in range(0, recording_data.shape[0], window_stride):
        
        if i + window_length > recording_data.shape[0]:
            break
        
        X_input.append(recording_data.iloc[i : i + window_length].to_numpy())
    
    X_input = np.stack(X_input)
    
    conv1 = tf.keras.models.Model(inputs = keras_model.layers[0].input,
                                  outputs = keras_model.layers[2].output)
    
    conv2 = tf.keras.models.Model(inputs = keras_model.layers[3].input,
                                  outputs = keras_model.layers[5].output)
    
    conv3 = tf.keras.models.Model(inputs = keras_model.layers[6].input,
                                  outputs = keras_model.layers[8].output)
    
    conv4 = tf.keras.models.Model(inputs = keras_model.layers[9].input,
                                  outputs = keras_model.layers[11].output)
    
    conv5 = tf.keras.models.Model(inputs = keras_model.layers[12].input,
                                  outputs = keras_model.layers[14].output)
    
    conv6 = tf.keras.models.Model(inputs = keras_model.layers[15].input,
                                  outputs = keras_model.layers[17].output)
    
    inference = tf.keras.models.Model(inputs = keras_model.layers[18].input,
                                  outputs = keras_model.layers[-1].output)
    
    samples_indexes = np.arange(1, X_input.shape[0])
    
    sample_shift = window_stride
    
    with tf.device("/:cpu0"):
        y_pred = keras_model.predict(X_input)
        
        prev_act_1 = conv1(X_input[:1, ...])
        prev_act_2 = conv2(prev_act_1)
        prev_act_3 = conv3(prev_act_2)
        prev_act_4 = conv4(prev_act_3)
        prev_act_5 = conv5(prev_act_4)
        prev_act_6 = conv6(prev_act_5)
        
        y_pred_shift = np.zeros((samples_indexes.size + 1, 2))
        y_pred_shift[0] = y_pred[0]
        
        for i in tqdm(samples_indexes):
            curX = X_input[i:i+1, ...]
            
            output_tmp1 = conv1(curX)
            output1 = tf.concat([prev_act_1[:, sample_shift:, :], 
                                  output_tmp1[:, -sample_shift:, :]], axis = 1)
            
          
            output_tmp2 = conv2(output1)
            output2 = tf.concat([prev_act_2[:, sample_shift:, :], 
                                  output_tmp2[:, -sample_shift:, :]], axis = 1)
            
          
            
            output_tmp3 = conv3(output2)
            output3 = tf.concat([prev_act_3[:, sample_shift:, :], 
                                  output_tmp3[:, -sample_shift:, :]], axis = 1)
            
            output_tmp4 = conv4(output3)
            output4 = tf.concat([prev_act_4[:, sample_shift:, :], 
                                  output_tmp4[:, -sample_shift:, :]], axis = 1)
            
            output_tmp5 = conv5(output4)
            output5 = tf.concat([prev_act_5[:, sample_shift:, :], 
                                  output_tmp5[:, -sample_shift:, :]], axis = 1)
            
            output_tmp6 = conv6(output5)
            output6 = tf.concat([prev_act_6[:, sample_shift:, :], 
                                  output_tmp6[:, -sample_shift:, :]], axis = 1)
            
                
            
            prev_act_1 = output1
            prev_act_2 = output2
            prev_act_3 = output3
            prev_act_4 = output4
            prev_act_5 = output5
            prev_act_6 = output6
            
            
            y_pred_shift[i] = inference(output6).numpy()
    
    results = {'y_pred' : y_pred,
               'y_pred_shift' : y_pred_shift}   
    
    output_file_name = './results/' + center + '/patient_' + str(patient)\
        + '/' 
        
    if not os.path.exists(output_file_name):
        os.makedirs(output_file_name)
        
    output_file_name += recording + '.pickle'
        
    with open(output_file_name, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
 
    