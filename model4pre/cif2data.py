import json
import warnings
import numpy as np
import pymatgen.core as mg
from ase.io import read,write
# from ase.io.cif import read_cif, write_cif
from pymatgen.io.ase import AseAtomsAdaptor
from ase import Atoms
from pymatgen.core import Structure
from pymatgen.io.cif import CifParser
from collections import defaultdict

def n_atom(mof):
    try:
        structure = read(mof)
        struct = AseAtomsAdaptor.get_structure(structure)
    except:
        struct = CifParser(mof, occupancy_tolerance=10)
        struct.get_structures()
    elements = [str(site.specie) for site in structure.sites]
    return len(elements)

def ase_format(mof):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            # mof_temp = Structure.from_file(mof,primitive=True)
            mof_temp = Structure.from_file(mof)
            mof_temp.to(filename=mof, fmt="cif")
            struc = read(mof)
            write(mof, struc)
            # print('Reading by ase: ' + mof)
    except:
        try:
            struc = read(mof)
            write(mof, struc)
            print('Reading by ase: ' + mof)
        except:
            print("An error occurred while reading: " + mof)

periodic_table_symbols = [
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg',
    'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr',
    'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
    'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf',
    'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po',
    'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm',
    'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs',
    'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'
    ]

def CIF2json(mof, save_path):
    try:
        structure = read(mof)
        struct = AseAtomsAdaptor.get_structure(structure)
    except:
        struct = CifParser(mof, occupancy_tolerance=10)
        struct.get_structures()
    _c_index, _n_index, _, n_distance = struct.get_neighbor_list(r=6, numerical_tol=0, exclude_self=True)
    _nonmax_idx = []
    for i in range(len(structure)):
        idx_i = (_c_index == i).nonzero()[0]
        idx_sorted = np.argsort(n_distance[idx_i])[: 200]
        _nonmax_idx.append(idx_i[idx_sorted])
    _nonmax_idx = np.concatenate(_nonmax_idx)
    index1 = _c_index[_nonmax_idx]
    index2 = _n_index[_nonmax_idx]
    dij = n_distance[_nonmax_idx]
    numbers = []
    elements = [str(site.specie) for site in struct.sites]
    for i in range(len(elements)):
        ele = elements[i]
        atom_index = periodic_table_symbols.index(ele)
        numbers.append(int(int(atom_index)+1))
    nn_num = []
    for i in range(len(structure)):
        j = 0
        for idx in range(len(index1)):
            if index1[idx] == i:
                    j += 1
            else:
                    pass
        nn_num.append(j)
    data = {"rcut": 6.0,
            "numbers": numbers,
            "index1": index1.tolist(),
            "index2":index2.tolist(),
            "dij": dij.tolist(),
            "nn_num": nn_num}
    name = mof.split('.cif')[0]
    return data

def pre4pre(mof, save_cell_dir, save_pos_dir):
    name = mof.split('.cif')[0]
    try:
        try:
            structure = mg.Structure.from_file(mof)
        except:
            try:
                atoms = read(mof)
                structure = AseAtomsAdaptor.get_structure(atoms)
            except:
                structure = CifParser(mof, occupancy_tolerance=10)
                structure.get_structures()
        coords = structure.frac_coords
        elements = [str(site.specie) for site in structure.sites]
        pos = []
        lattice = structure.lattice.matrix
        for i in range(len(elements)):
            x = coords[i][0]
            y = coords[i][1]
            z = coords[i][2]
            pos.append([float(x),float(y),float(z)])
    except Exception as e:
        pass
    return lattice, pos

def average_and_replace(numbers, di):
    groups = defaultdict(list)
    for i, number in enumerate(numbers):
        if di ==3:
            key = format(number, '.3f')
            groups[key].append(i)
        elif di ==2:
            key = format(number, '.2f')
            groups[key].append(i)
    for key, indices in groups.items():
        avg = sum(numbers[i] for i in indices) / len(indices)
        for i in indices:
            numbers[i] = avg
    return numbers

def extract_charges_from_cif(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    start_reading_charges = False
    atoms = []
    for line in lines:
        if start_reading_charges:
            try:
                atom = line.split()[0]
                atoms.append(atom)
            except ValueError:
                continue
        if "_atom_site_charge" in line:
            start_reading_charges = True
    return atoms

def write4cif(name, chg, digits, atom_type_option, neutral_option, charge=False):
    new_content = []
    dia = int(digits)
    if charge:
        if atom_type_option:
            gcn_charge = chg.numpy()
            sum_chg = sum(gcn_charge)
            if neutral_option:
                charge = average_and_replace(gcn_charge,di=3)
                sum_chg = sum(charge)
                charges_1 = []
                for c in charge:
                    cc = c - sum_chg/len(charge)
                    charges_1.append(round(cc, dia))
                charge_2 = average_and_replace(charges_1,di=2)
                sum_chg = sum(charge_2)
                charges = []
                for c in charge_2:
                    cc = c - sum_chg/len(charge_2)
                    charges.append(round(cc, dia))
            else:
                charge = average_and_replace(gcn_charge,di=3)
                charges_1 = []
                for c in charge:
                    charges_1.append(round(c, dia))
                charge_2 = average_and_replace(charges_1,di=2)
                charges = []
                for c in charge_2:
                    charges.append(round(c, dia))
            net_charge = sum(charges)
            atoms = extract_charges_from_cif(name + ".cif")
            unique_counts = {}
            for char, index in zip(atoms, charges):
                if char not in unique_counts:
                    unique_counts[char] = set()  # Use a set to automatically handle uniqueness
                unique_counts[char].add(index)
            result = {char: len(indices) for char, indices in unique_counts.items()}
            atom_type = result
        else:
            gcn_charge = chg.numpy()
            sum_chg = sum(gcn_charge)
            if neutral_option:
                charges = [round(c - sum_chg/len(gcn_charge), dia) for c in gcn_charge]
            else:
                charges = []
                for c in gcn_charge:
                    charges.append(round(c, dia)
            net_charge = sum(charges)
            atom_type = "Failure to check like atoms"
        with open(name + ".cif", 'r') as file:
            lines = file.readlines()
        new_content.append("# generated by PACMAN, DOI: *** \n")
        new_content.append("data_structure\n")
        charge_inserted = False
        charge_index = 0
        for line in lines:
            # Replace specific lines
            line = line.replace('_space_group_name_H-M_alt', '_symmetry_space_group_name_H-M')
            line = line.replace('_space_group_IT_number', '_symmetry_Int_Tables_number')
            line = line.replace('_space_group_symop_operation_xyz', '_symmetry_equiv_pos_as_xyz')
            if '_atom_site_occupancy' in line and not charge_inserted:
                new_content.append(line)
                new_content.append("  _atom_site_charge\n")
                charge_inserted = True
            elif charge_inserted and charge_index < len(charges):
                new_content.append(line.strip() + " " + str(charges[charge_index]) + "\n")
                charge_index += 1
            else:
                new_content.append(line)
    return ''.join(new_content),atom_type,net_charge
