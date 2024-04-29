import torch
import pickle
from model4pre.GCN_ddec import SemiFullGN
from model4pre.data import collate_pool, get_data_loader, CIFData, load_gcn
from model4pre.cif2data import ase_format, CIF2json, pre4pre, write4cif
import importlib
import sys

sys.modules['source'] = importlib.import_module('model4pre')
sys.modules['GCNCharge'] = importlib.import_module('model4pre')
sys.modules['model.utils'] = importlib.import_module('model4pre.utils')
sys.modules['source.utils'] = importlib.import_module('model4pre.utils')
sys.modules['model'] = importlib.import_module('model4pre')


def predict_with_model(model_name, charge_name, file,name, digits, atom_type_option, neutral_option):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_pbe_name = "./pth/MOF-PBE/pbe-atom.pth"
    if model_name == "COF":
        charge_model_name = "./pth/COF/ddec.pth"
        nor_name = "./pth/COF/normalizer-ddec.pkl"
    else:
        if charge_name=="DDEC":
            charge_model_name = "./pth/MOF-DDEC/ddec.pth"
            nor_name = "./pth/MOF-DDEC/normalizer-ddec.pkl"
        elif charge_name=="Bader":
            charge_model_name = "./pth/MOF-Bader/bader.pth"
            nor_name = "./pth/MOF-Bader/normalizer-bader.pkl"
        elif charge_name=="CM5":
            charge_model_name = "./pth/MOF-CM5/cm5.pth"
            nor_name = "./pth/MOF-CM5/normalizer-cm5.pkl"
    gcn = load_gcn(model_pbe_name)
    with open(nor_name, 'rb') as f:
        ddec_nor = pickle.load(f)
    f.close()
    ase_format(file)
    data = CIF2json(file)
    lattice, pos = pre4pre(file)
    batch_size = 1
    num_workers = 0
    pin_memory = False
    pre_dataset = CIFData(data,lattice,pos)
    collate_fn = collate_pool
    pre_loader= get_data_loader(pre_dataset,collate_fn,batch_size,num_workers,pin_memory)
    structures, _, _ = pre_dataset[0]
    chg_1 = structures[0].shape[-1] + 3
    chg_2 = structures[1].shape[-1]
    chkpt_ddec = torch.load(charge_model_name, map_location=torch.device(device))
    model4chg = SemiFullGN(chg_1,chg_2,128,8,256)
    model4chg.cuda() if torch.cuda.is_available() else model4chg.to(device)
    model4chg.load_state_dict(chkpt_ddec['state_dict'])
    model4chg.eval()
    for _, (input,_) in enumerate(pre_loader):
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
            chg = chg.data.cpu()
            chg = ddec_nor.denorm(chg.data.cpu())
            result,atom_type_count,net_charge = write4cif(name, chg, digits, atom_type_option, neutral_option, charge_name)
    return result,atom_type_count,net_charge 
