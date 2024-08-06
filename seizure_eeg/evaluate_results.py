import pickle
import numpy as np

# Results for StreamiNNC

nrmse_ch0s = np.zeros(3)
nrmse_ch1s = np.zeros(3)

for i in range(1, 4):
    folder = 'Global_Gr' + str(i)
    with open('./results/' + folder +'/errors.pkl', 'rb') as handle:
        b = pickle.load(handle)

    nrmse_ch0s[i - 1] = b['nrmse_ch0s'].mean()
    nrmse_ch1s[i - 1] = b['nrmse_ch1s'].mean()

print("Streaming Inference")
print ('Output 0: ', nrmse_ch0s.mean() * 100)
print ('Output 1: ', nrmse_ch1s.mean() * 100)


## Results for Approx. StreamiNNC

nrmse_ch0s = np.zeros(3)
nrmse_ch1s = np.zeros(3)

for i in range(1, 4):
    folder = 'Global_Gr' + str(i)
    with open('./results_approx/' + folder +'/errors.pkl', 'rb') as handle:
        b = pickle.load(handle)

    nrmse_ch0s[i - 1] = b['nrmse_ch0s'].mean()
    nrmse_ch1s[i - 1] = b['nrmse_ch1s'].mean()
    
    
print("===================")
print("===================")
print("Approximate Streaming Inference")
print ('Output 0: ', nrmse_ch0s.mean() * 100)
print ('Output 1: ', nrmse_ch1s.mean() * 100)
