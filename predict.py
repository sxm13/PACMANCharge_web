import torch
import pickle
from model4pre.GCN_E import GCN
from model4pre.GCN_charge import SemiFullGN
from model4pre.data import collate_pool, get_data_loader, CIFData
from model4pre.cif2data import ase_format, CIF2json, pre4pre, write4cif

def predict_with_model(charge_name, file, name, digits, atom_type_option, neutral_option, connect_option):

    result = None
    atom_type_count = None
    net_charge = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if charge_name=="DDEC6":
        charge_model_name = "./pth/best_ddec/ddec.pth"
        nor_name = "./pth/best_ddec/normalizer-ddec.pkl"
    elif charge_name=="Bader":
        charge_model_name = "./pth/best_bader/bader.pth"
        nor_name = "./pth/best_bader/normalizer-bader.pkl"
    elif charge_name=="CM5":
        charge_model_name = "./pth/best_cm5/cm5.pth"
        nor_name = "./pth/best_cm5/normalizer-cm5.pkl"
    elif charge_name=="REPEAT":
        charge_model_name = "./pth/best_repeat/repeat.pth"
        nor_name = "./pth/best_repeat/normalizer-repeat.pkl"
    with open(nor_name, 'rb') as f:
        charge_nor = pickle.load(f)
    f.close()
    if connect_option:
        pass
    else:
        ase_format(file)
    data = CIF2json(file)
    pos = pre4pre(file)
    batch_size = 1
    num_workers = 0
    pin_memory = False
    pre_dataset = CIFData(data,pos,6,0,0.2)
    collate_fn = collate_pool
    pre_loader= get_data_loader(pre_dataset,collate_fn,batch_size,num_workers,pin_memory)
    for batch in pre_loader:
        chg_1 = batch[0].shape[-1]+3
        chg_2 = batch[1].shape[-1]
    gcn = GCN(chg_1-3, chg_2, 128, 7, 256,5) 
    chkpt = torch.load(charge_model_name, map_location=torch.device(device))
    model4chg = SemiFullGN(chg_1,chg_2,128,8,256)
    model4chg.to(device)
    model4chg.load_state_dict(chkpt['state_dict'])
    model4chg.eval()
    for _, (input) in enumerate(pre_loader):
        with torch.no_grad():
            input_var = (input[0].to(device),
                        input[1].to(device),
                        input[2].to(device),
                        input[3].to(device),
                        input[4].to(device),
                        input[5].to(device))
            encoder_feature = gcn.Encoding(*input_var)
            atoms_fea = torch.cat((input[0],input[7]),dim=-1)
            input_var2 = (atoms_fea.to(device),
                        input[1].to(device),
                        input[2].to(device),
                        input[3].to(device),
                        input[4].to(device),
                        input[5].to(device),
                        encoder_feature.to(device))
            chg = model4chg(*input_var2)
            charge_pre = charge_nor.denorm(chg.data.cpu())
            result,atom_type_count,net_charge = write4cif(name, charge_pre, digits, atom_type_option, neutral_option, charge_name, connect_option)
    return result,atom_type_count,net_charge 
