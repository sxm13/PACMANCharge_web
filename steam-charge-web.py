import streamlit as st
import time
start_time = time.perf_counter()
import os
import json
import torch
import pickle
import numpy as np
from model4pre.GCN_ddec import SemiFullGN
from model4pre.data import collate_pool, get_data_loader, CIFData, load_gcn
from model4pre.cif2data import ase_format, CIF2json, pre4pre, write4cif

def predict_with_model(model_name, file):
    model_pbe_name = "./pth/best_pbe/pbe-atom.pth"
    if model_name == "COF":
        model_ddec_name = "./pth/best_ddec_COF/ddec.pth"
        ddec_nor_name = "./pth/best_ddec_COF/normalizer-ddec.pkl"
    else:
        model_ddec_name = "./pth/best_ddec/ddec.pth"
        ddec_nor_name = "./pth/best_ddec/normalizer-ddec.pkl"
    gcn = load_gcn(model_pbe_name)
    with open(ddec_nor_name, 'rb') as f:
        ddec_nor = pickle.load(f)
    f.close()
    ase_format(file)
    CIF2json(file,save_path="")
    pre4pre(file,"","")
    batch_size = 1
    num_workers = 0
    pin_memory = False
    pre_dataset = CIFData(file,"","")
    collate_fn = collate_pool
    pre_loader= get_data_loader(pre_dataset,collate_fn,batch_size,num_workers,pin_memory)
    structures, _, _ = pre_dataset[0]

    chg_1 = structures[0].shape[-1] + 3
    chg_2 = structures[1].shape[-1]
    chkpt_ddec = torch.load(model_ddec_name, map_location=torch.device(device))
    model4chg = SemiFullGN(chg_1,chg_2,128,8,256)
    model4chg.cuda() if torch.cuda.is_available() else model4chg.to(device)
    model4chg.load_state_dict(chkpt_ddec['state_dict'])
    model4chg.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for _, (input,cif_ids) in enumerate(pre_loader):
        with torch.no_grad():
            if device == "cuda":
                input_cuda = [input_tensor.to(device) for input_tensor in input]
                input_var = (input_cuda[0].cuda(),
                            input_cuda[1].cuda(),
                            input_cuda[2].cuda(),
                            input_cuda[3].cuda(),
                            input_cuda[4].cuda(),
                            input_cuda[5].cuda())
                relaxed_feature = gcn.Encoding(*input_var)
                atoms_fea = torch.cat((input_cuda[0],input_cuda[7]),dim=-1)
                input_var2 = (atoms_fea.cuda(),
                        input_cuda[1].cuda(),
                        input_cuda[2].cuda(),
                        input_cuda[3].cuda(),
                        input_cuda[4].cuda(),
                        input_cuda[5].cuda(),
                        relaxed_feature.cuda(),
                        input_cuda[9][:,:9].cuda())
            else:
                input_var = (input[0],
                            input[1],
                            input[2],
                            input[3],
                            input[4],
                            input[5])
                relaxed_feature = gcn.Encoding(*input_var)
                atoms_fea = torch.cat((input[0],input[7]),dim=-1)
                input_var2 = (atoms_fea,
                        input[1],
                        input[2],
                        input[3],
                        input[4],
                        input[5],
                        relaxed_feature,
                        input[9][:,:9])
            chg = model4chg(*input_var2)
            chg = ddec_nor.denorm(chg.data.cpu())
            name = cif_ids[0]+'_charge.npy'
            np.save(""+name,chg)
            result = write4cif(file,"","","","",charge = True)
            os.remove(cif_ids[0] + '.json')
            os.remove(cif_ids[0] + '_cell.npy')
            os.remove(cif_ids[0] + '_pos.npy')
            os.remove(cif_ids[0] + '_charge.npy')
    return result

st.title('MOF/COF Charges predicter')
uploaded_file = st.file_uploader("Please upload your CIF file", type="cif")
model_option = st.radio("Type", ('MOF', 'COF'))

if uploaded_file is not None and model_option:
    prediction = predict_with_model(model_option, uploaded_file)
    st.write("predicted result: ")
    st.write(prediction)
    st.download_button(label="Download cif file with charges", data=prediction, file_name="prediction.cif", mime="text/cif")
