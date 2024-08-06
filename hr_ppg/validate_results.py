import numpy as np
import scipy
import pickle

def get_nrmse(output, stream_output):
    error = output - stream_output
    nrmse = np.sqrt(np.mean(error**2)) / (np.max(output) - np.min(output))
    
    return nrmse

# ZERO PADDING
zero_padding_nrmse = np.zeros((15,))
zero_padding_full_inference_mae = np.zeros((15))
zero_padding_shift_inference_mae = np.zeros((15,))
zero_padding_pearson = np.zeros((15,))
for test_subject_id in range(1, 16):
    with open('./results/zero_padding/S' \
              + str(test_subject_id) \
                  +'.pickle', 'rb') as handle:
        b = pickle.load(handle)
    
    zero_padding_full_inference_mae[test_subject_id - 1] = np.mean(np.abs(b['y_test'] - b['y_pred']))
    zero_padding_shift_inference_mae[test_subject_id - 1] = np.mean(np.abs(b['y_test'] - b['y_pred_shift']))
    zero_padding_nrmse[test_subject_id - 1] = get_nrmse(b['y_pred'], b['y_pred_shift'])
    zero_padding_pearson = np.corrcoef(b['y_pred'], b['y_pred_shift'])[0, 1]

print("== ZERO PADDING ==")
print("% NRMSE: ", zero_padding_nrmse.mean() * 100)
print("% Pearson: ", zero_padding_pearson.mean() * 100)
print("MAE (full): ", zero_padding_full_inference_mae.mean())
print("MAE (streaming): ", zero_padding_shift_inference_mae.mean())

print("")

# SIGNAL PADDING
signal_padding_nrmse = np.zeros((15,))
signal_padding_full_inference_mae = np.zeros((15))
signal_padding_shift_inference_mae = np.zeros((15,))
signal_padding_pearson = np.zeros((15,))
for test_subject_id in range(1, 16):
    with open('./results/signal_padding/S' \
              + str(test_subject_id) \
                  +'.pickle', 'rb') as handle:
        b = pickle.load(handle)
    
    signal_padding_full_inference_mae[test_subject_id - 1] = np.mean(np.abs(b['y_test'][4:] - b['y_pred']))
    signal_padding_shift_inference_mae[test_subject_id - 1] = np.mean(np.abs(b['y_test'] - b['y_pred_shift']))
    signal_padding_nrmse[test_subject_id - 1] = get_nrmse(b['y_pred'], b['y_pred_shift'][4:])
    signal_padding_pearson[test_subject_id - 1] = np.corrcoef(b['y_pred'], b['y_pred_shift'][4:])[0, 1]

print("== SIGNAL PADDING ==")
print("% NRMSE: ", signal_padding_nrmse.mean() * 100)
print("% Pearson: ", signal_padding_pearson.mean() * 100)
print("MAE (full): ", signal_padding_full_inference_mae.mean())
print("MAE (streaming): ", signal_padding_shift_inference_mae.mean())
print("")

# ZERO PADDING APPROXIMATE
zero_padding_approximate_nrmse = np.zeros((15,))
zero_padding_approximate_full_inference_mae = np.zeros((15))
zero_padding_approximate_shift_inference_mae = np.zeros((15,))
zero_padding_approximate_pearson = np.zeros((15,))
for test_subject_id in range(1, 16):
    with open('./results/zero_padding_approximate/S' \
              + str(test_subject_id) \
                  +'.pickle', 'rb') as handle:
        b = pickle.load(handle)
    
    zero_padding_approximate_full_inference_mae[test_subject_id - 1] = np.mean(np.abs(b['y_test'] - b['y_pred']))
    zero_padding_approximate_shift_inference_mae[test_subject_id - 1] = np.mean(np.abs(b['y_test'] - b['y_pred_shift']))
    zero_padding_approximate_nrmse[test_subject_id - 1] = get_nrmse(b['y_pred'], b['y_pred_shift'])
    zero_padding_approximate_pearson[test_subject_id - 1] = np.corrcoef(b['y_pred'], b['y_pred_shift'])[0, 1]

print("== ZERO PADDING APPROXIMATE ==")
print("% NRMSE: ", zero_padding_approximate_nrmse.mean() * 100)
print("% Pearson: ", zero_padding_approximate_pearson.mean() * 100)
print("MAE (full): ", zero_padding_approximate_full_inference_mae.mean())
print("MAE (streaming): ", zero_padding_approximate_shift_inference_mae.mean())
print("")

# ZERO PADDING POOL MISALIGNMENT
zero_padding_pm_nrmse = np.zeros((15,))
zero_padding_pm_full_inference_mae = np.zeros((15))
zero_padding_pm_shift_inference_mae = np.zeros((15,))
for test_subject_id in range(1, 16):
    with open('./results/zero_padding_pooling_misalignment/S' \
              + str(test_subject_id) \
                  +'.pickle', 'rb') as handle:
        b = pickle.load(handle)
    
    zero_padding_pm_full_inference_mae[test_subject_id - 1] = np.mean(np.abs(b['y_test'] - b['y_pred']))
    zero_padding_pm_shift_inference_mae[test_subject_id - 1] = np.mean(np.abs(b['y_test'] - b['y_pred_shift']))
    zero_padding_pm_nrmse[test_subject_id - 1] = get_nrmse(b['y_pred'], b['y_pred_shift'])

print("== ZERO PADDING POOL MISALIGNMENT==")
print("% NRMSE: ", zero_padding_pm_nrmse.mean() * 100)
print("MAE (full): ", zero_padding_pm_full_inference_mae.mean())
print("MAE (streaming): ", zero_padding_pm_shift_inference_mae.mean())
print("")

# ZERO PADDING PARTIAL STREAMING
zero_padding_partial_streaming_nrmse = np.zeros((15,))
zero_padding_partial_streaming_full_inference_mae = np.zeros((15))
zero_padding_partial_streaming_shift_inference_mae = np.zeros((15,))
zero_padding_partial_streaming_pearson = np.zeros((15,))

for test_subject_id in range(1, 16):
    with open('./results/zero_padding_partial_streaming/S' \
              + str(test_subject_id) \
                  +'.pickle', 'rb') as handle:
        b = pickle.load(handle)
    
    zero_padding_partial_streaming_full_inference_mae[test_subject_id - 1] = np.mean(np.abs(b['y_test'] - b['y_pred']))
    zero_padding_partial_streaming_shift_inference_mae[test_subject_id - 1] = np.mean(np.abs(b['y_test'] - b['y_pred_shift']))
    zero_padding_partial_streaming_nrmse[test_subject_id - 1] = get_nrmse(b['y_pred'], b['y_pred_shift'])
    zero_padding_partial_streaming_pearson[test_subject_id - 1] = np.corrcoef(b['y_pred'], b['y_pred_shift'])[0, 1]

print("== ZERO PADDING PPARTIAL STREAMING ==")
print("% NRMSE: ", zero_padding_partial_streaming_nrmse.mean() * 100)
print("% Pearson: ", zero_padding_partial_streaming_pearson.mean() * 100)
print("MAE (full): ", zero_padding_partial_streaming_full_inference_mae.mean())
print("MAE (streaming): ", zero_padding_partial_streaming_shift_inference_mae.mean())

