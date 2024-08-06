import os
import sys
import torch
import pandas as pd
import pickle
from torch.utils.data import DataLoader
from utils import FCN, StreamFCN, chbmit_test

from tqdm import tqdm
import numpy as np
# Handle later what will happen for the "global"

def get_nrmse(output, stream_output):
    error = output - stream_output
    nrmse = np.sqrt(np.mean(error**2)) / (np.max(output) - np.min(output))
    
    return nrmse

def test(test_loader, model, stream_model, device):

    nrmse_ch0s = []
    nrmse_ch1s = []
    pc_ch0s = []
    pc_ch1s = []
    for i, (data, target) in enumerate(test_loader):
        
        # Batch inference
        data, target = data.float().to(device), target.to(device)
        output = model(data)
        output = output.detach().numpy()

        # Stream inference
        stream_output = torch.zeros((data.shape[0], 2))
        stream_output[:1, ...] = stream_model.init_buffer(data[:1])        
        for j in tqdm(range(1, data.shape[0]), ascii = True):
            stream_output[j:j + 1, :] = stream_model(data[j:j+1, ...])
        stream_output = stream_output.detach().numpy()

        nrmse_ch0 = get_nrmse(output[:, 0], stream_output[:, 0])
        nrmse_ch1 = get_nrmse(output[:, 1], stream_output[:, 1])

        nrmse_ch0s.append(nrmse_ch0)
        nrmse_ch1s.append(nrmse_ch1)
        pc_ch0s.append(np.corrcoef(output[:, 0], stream_output[:, 0])[0, 1])
        pc_ch1s.append(np.corrcoef(output[:, 1], stream_output[:, 1])[0, 1])
    
    nrmse_ch0s = np.array(nrmse_ch0s)
    nrmse_ch1s = np.array(nrmse_ch1s)
    pc_ch0s = np.array(pc_ch0s)
    pc_ch1s = np.array(pc_ch1s)

    return nrmse_ch0s, nrmse_ch1s, pc_ch0s, pc_ch1s


def load_pat_test_data(pat_dir):
    test_dir = os.path.join(pat_dir, "test.pkl")

    with open(test_dir, "rb") as file:
        test_dataset = pickle.load(file)

    test_dataset = chbmit_test.CHB_Store_Test.CHBMITest.from_pkl(test_dataset)
    info_test = test_dataset.seg_list[:, :-1]
    test_df = pd.DataFrame(
        info_test,
        columns=["file_id", "start_time", "end_time", "true_label"],
    )
    test_loader = DataLoader(
        test_dataset, batch_size=len(test_dataset), shuffle=False
    )
    return test_loader, test_df

def eval_global_exp(g_m_dir, ch_dir, out_dir, patients, map_location, device):
    model = FCN.Net(in_channels=18).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(device)
    try:
        params = torch.load(g_m_dir, map_location=map_location)
    except FileNotFoundError:
        print("The specified file was not found.")
        return
    except RuntimeError as e:
        print(f"An error occurred while loading the model: {e}")
        return
    model.load_state_dict(params["state_dict"])
    model.eval()

    stream_model = StreamFCN.ApproximateStreamNet(in_channels=18, overlap = 256).to(device)
    stream_model.load_state_dict(params["state_dict"])
    stream_model.eval()

    all_patient_nrmse_ch0s = []
    all_patient_nrmse_ch1s = []
    all_pc_ch0s = []
    all_pc_ch1s = []
    for i, pat in enumerate(patients):
        # data directories
        pat_dir = os.path.join(ch_dir, pat)
        fld_dirs = sorted(os.listdir(pat_dir), key=lambda x: int(x))
        data_dirs = [os.path.join(pat_dir, item) for item in fld_dirs]

        label_dir = os.path.join(out_dir, pat[-2:])
        if not os.path.exists(label_dir):
            os.mkdir(label_dir)

        label_dir_t = os.path.join(label_dir, "0")
        if not os.path.exists(label_dir_t):
            os.mkdir(label_dir_t)

        patient_all_nrmse_ch0s = []
        patient_all_nrmse_ch1s = []
        pc_all_ch0s = []
        pc_all_ch1s = []
        for n, data_dir in enumerate(data_dirs):
            print(data_dir)
            test_loader, df = load_pat_test_data(data_dir)

            discontinuity_count = np.sum(np.diff(df['start_time']) > 1)
            if discontinuity_count > 0:
                continue

            nrmse_ch0s, nrmse_ch1s, pc_ch0s, pc_ch1s = test(test_loader, model, stream_model, device)
            
            patient_all_nrmse_ch0s.append(nrmse_ch0s.mean())
            patient_all_nrmse_ch1s.append(nrmse_ch1s.mean())
            pc_all_ch0s.append(pc_ch0s.mean())
            pc_all_ch1s.append(pc_ch1s.mean())
            
        
        patient_all_nrmse_ch0s = np.array(patient_all_nrmse_ch0s)
        patient_all_nrmse_ch1s = np.array(patient_all_nrmse_ch1s)
        pc_all_ch0s = np.array(pc_all_ch0s)
        pc_all_ch1s = np.array(pc_all_ch1s)


        all_patient_nrmse_ch0s.append(patient_all_nrmse_ch0s.mean())
        all_patient_nrmse_ch1s.append(patient_all_nrmse_ch1s.mean())
        all_pc_ch0s.append(pc_all_ch0s.mean())
        all_pc_ch1s.append(pc_all_ch1s.mean())
        
    all_patient_nrmse_ch0s = np.array(all_patient_nrmse_ch0s)
    all_patient_nrmse_ch1s = np.array(all_patient_nrmse_ch1s)
    all_pc_ch0s = np.array(all_pc_ch0s)
    all_pc_ch1s = np.array(all_pc_ch1s)

    results = {
        'nrmse_ch0s' : all_patient_nrmse_ch0s,
        'nrmse_ch1s' : all_patient_nrmse_ch1s,
        'pc_ch0s' : all_pc_ch0s,
        'pc_ch1s' : all_pc_ch1s
    }
    

    with open(out_dir + '/errors.pkl', 'wb') as f:
        pickle.dump(results, f)


def main():
    if "--quiet" in sys.argv or "-q" in sys.argv:

        class Quiet:
            def write(self, *args, **kwargs):
                pass

        sys.stdout = Quiet()  # Prevents the script from printing

    cuda = torch.cuda.is_available()
    # device = torch.device("cuda:0" if cuda else "cpu")
    device = "cpu"
    print("Device: {}".format(device))

    torch.manual_seed(0)
    if cuda:
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(0)
        map_location = None
    else:

        def map_location(storage, loc):
            return storage

    exps_name = ["Global"]
    groups = [1, 2, 3]

    group_1 = [6, 21, 9, 7, 4, 23, 5, 8]
    group_2 = [2, 22, 20, 18, 3, 10, 11, 12]
    group_3 = [16, 14, 19, 17, 13, 1, 24, 15]

    out_folder = "results_approx"
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    ch_dir = "./personalized-online-seizure-detection-dev/test_chunks"
    base_exp_dir = "experiments_out"
    all_patients = ["chb" + str(i).zfill(2) for i in range(1, 25)]
    global_model_dir_base = './model/global_models'
    # Generating csv files for each experiment
    for exp in exps_name:
        for gr in groups:
            if gr == 1:
                group_client = group_2 + group_3
                global_mdl_name = "Global_Gr1.pth"
            elif gr == 2:
                group_client = group_1 + group_3
                global_mdl_name = "Global_Gr2.pth"
            elif gr == 3:
                group_client = group_1 + group_2
                global_mdl_name = "Global_Gr3.pth"
            patients = [all_patients[i - 1] for i in group_client]

            global_mdl_dir = os.path.join(
                global_model_dir_base, global_mdl_name
            )
            out_exp = os.path.join(out_folder, global_mdl_name[:-4])
            if not os.path.exists(out_exp):
                os.mkdir(out_exp)

            eval_global_exp(
                global_mdl_dir,
                ch_dir,
                out_exp,
                patients,
                map_location,
                device,
            )
                


if __name__ == "__main__":
    main()
