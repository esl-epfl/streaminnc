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

    
batch_size = 1

fs = 32.0

# load a time-serie
data_path = Path("./gtcs-detection-model-export/seizure-subset/")

center = "dianalund"
patient = 19
recording = "1597068172_A0249F"

#----------------------------------------------------------------------------------------------------------------------#
# load gtcs info
gtcs_df = pd.read_csv(data_path / "gtcs_subset.csv")
gtcs_df = gtcs_df[(gtcs_df["center"] == center) & (gtcs_df["patient"] == patient) & (gtcs_df["E4_recording"] == recording)]

# ensure the used columns are in the correct format
gtcs_df['focal_onset'] = pd.to_datetime(gtcs_df['focal_onset'])
gtcs_df['clonic_end'] = pd.to_datetime(gtcs_df['clonic_end'])

# load recording
recording_data = pd.read_csv(data_path / center / f"patient_{patient}" / f"{recording}_ACC.parquet",)


window_length = 960
window_stride = 160 
X_input = []
for i in range(0, recording_data.shape[0], window_stride):
    
    if i + window_length > recording_data.shape[0]:
        break
    
    X_input.append(recording_data.iloc[i : i + window_length].to_numpy())

X_input = np.stack(X_input)

keras_model = define_model()
keras_model.summary()

output_test_file = './example_data/test_data.h'
testId = 1
test_indexes = [5256, 18527]

with open(output_test_file, "w") as f:
    
    for index in test_indexes:
        current_sample = X_input[index]
        
        
        output_string = 'double input' + str(testId) + '[960][4] = {'
    
        for i in range(current_sample.shape[0]):
            output_string += "{"
            for j in range(current_sample.shape[1]):
                output_string +=  "{:.15f}".format(current_sample[i][j]) + ", "
            output_string += "},"
        output_string += "};\n\n"
        
        f.write(output_string)
        testId += 1
        
output_test_file = './example_data/test_data_multiple_samples.h'
testId = 1
test_index = 5256

field_of_view = 0

patch_size = window_stride
kernel_size = 3
strides = 1

window_size = patch_size
step_size = patch_size

with open(output_test_file, "w") as f:
    
    sample = X_input[test_index]
    
    test_indexes = np.arange(0, sample.shape[0], 160)
    output_string = 'double input' + str(testId) + '[6][960][4] = {'

    for l in test_indexes:
        current_sample = sample[l : l + 160, :]
        
        output_string += '{'
        for i in range(current_sample.shape[0]):
            output_string += "{"
            for j in range(current_sample.shape[1]):
                output_string +=  "{:.15f}".format(current_sample[i][j]) + ", "
            output_string += "},"
        output_string += "},"
        
    output_string += "};\n\n"
    
    f.write(output_string)
    testId += 1

output_folder = './saved_models/decomposed_layers/'

string = ""

layer_names = ['']

convId = 1
bnId = 1
denseId = 1

with open(output_folder + "weight_definitions.h", "w") as f:
    padding = '    '
    
    for layer in keras_model.layers:
        
        layer_name = layer.__class__.__name__
        
        if layer_name == 'Conv1D':
    
            weights = layer.get_weights()[0]
            weights = np.transpose(weights, axes = (2, 0, 1))
            print(weights.shape)
            
            s1 = weights.shape[0]
            s2 = weights.shape[1]
            s3 = weights.shape[2]
            
            output_string = "const int conv" + str(convId) + "_size1 = " + \
                str(s1) + ";\n\n"
            f.write(output_string)
            
            output_string = "const int conv" + str(convId) + "_size2 = " + \
                str(s2) + ";\n\n"
            f.write(output_string)
            
            output_string = "const int conv" + str(convId) + "_size3 = " + \
                str(s3) + ";\n\n"
            f.write(output_string)
            
            output_string = "const double conv" + str(convId) + "_weights[" + str(s1) + "]["\
                + str(s2) + "][" + str(s3) +"] = \n{"
            for i in range(weights.shape[0]):
                output_string += '{'
                for j in range(weights.shape[1]):
                    output_string += '{'
                    for k in range(weights.shape[2]):
                        output_string +=  "{:.15f}".format(weights[i, j, k]) + ','
                        
                    output_string += '},\n'
                output_string += '},\n'
            output_string += '};\n\n'
            
            f.write(output_string)
            
            bias = layer.get_weights()[1]
            output_string = "const double conv" + str(convId) + "_bias[" \
                + str(bias.shape[0]) + "] = \n{"
            
            for i in range(bias.shape[0]):
                output_string +=  "{:.15f}".format(bias[i]) + ','
            output_string += '};\n\n'
            
            f.write(output_string)
        
            
            convId += 1 

        if layer_name == 'Dense':
    
            weights = layer.get_weights()[0]
            # weights = np.transpose(weights)
            
            s1 = weights.shape[0]
            s2 = weights.shape[1]
            
            output_string = "const int dense" + str(denseId) + "_size1 = " + \
                str(s1) + ";\n\n"
            f.write(output_string)
            
            output_string = "const int dense" + str(denseId) + "_size2 = " + \
                str(s2) + ";\n\n"
            f.write(output_string)

            
            output_string = "const double dense" + str(denseId) + "_weights[" + str(s1) + "]["\
                + str(s2) + "] = \n{"
            for i in range(weights.shape[0]):
                output_string += '{'
                for j in range(weights.shape[1]):
                    output_string +=  "{:.15f}".format(weights[i, j]) + ','
                output_string += '},\n'
            output_string += '};\n\n'
            
            f.write(output_string)
            
            bias = layer.get_weights()[1]
            
            
            output_string = "const double dense" + str(denseId) + "_bias[" \
                + str(bias.shape[0]) + "] = \n{"
            
            for i in range(bias.shape[0]):
                output_string += "{:.15f}".format(bias[i]) + ","
            output_string += "};\n\n"
            
            f.write(output_string)
            
            denseId += 1 
            
        if layer_name == 'BatchNormalization':
            
            weights = layer.get_weights()
            
            gamma = weights[0]
            beta = weights[1]
            mean = weights[2]
            variance = weights[3]
            
            output_string = "const int bn" + str(bnId) + "_size = "\
                + str(gamma.shape[0]) + ";\n\n"
            f.write(output_string)
            
            output_string = "const double bn" + str(bnId) + "_gamma[" \
                + str(gamma.shape[0]) + "] = \n{"
            
            for i in range(gamma.shape[0]):
                output_string +=  "{:.15f}".format(gamma[i]) + ","
            output_string += "};\n\n"
            
            f.write(output_string)
            
            output_string = "const double bn" + str(bnId) + "_beta[" \
                + str(beta.shape[0]) + "] = \n{"
            
            for i in range(beta.shape[0]):
                output_string +=  "{:.15f}".format(beta[i]) + ","
            output_string += "};\n\n"
            
            f.write(output_string)
            
            output_string = "const double bn" + str(bnId) + "_mean[" \
                + str(mean.shape[0]) + "] = \n{"
            
            for i in range(mean.shape[0]):
                output_string +=  "{:.15f}".format(mean[i]) + ","
            output_string += "};\n\n"
            
            f.write(output_string)
            
            output_string = "const double bn" + str(bnId) + "_variance[" \
                + str(variance.shape[0]) + "] = \n{"
            
            for i in range(variance.shape[0]):
                output_string +=  "{:.15f}".format(variance[i]) + ","
            output_string += "};\n\n"
            
            f.write(output_string)
            
            
            
            bnId += 1
            