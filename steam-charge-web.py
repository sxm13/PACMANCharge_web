# import streamlit as st
# from predict import predict_with_model
# from stmol import *
# import py3Dmol
# from ase.io import read, write
# from io import StringIO

# st.markdown("""
#     <style>
#     .big-font {
#         font-size:30px !important;
#         color: #FF4B4B;
#     }
#     .blue-text {
#         color: #4F8BF9;
#     }
#     .green-text {
#         color: #49BE25;
#     }
#     .custom-button {
#         background-color: #FF4B4B;
#         color: white;
#         padding: 10px 24px;
#         border: none;
#         border-radius: 4px;
#         cursor: pointer;
#     }
#     .custom-button:hover {
#         background-color: #FF7878;
#     }
#     </style>
#     """, unsafe_allow_html=True)

# st.markdown('<h1 class="big-font">💭 MOF/COF GCN Charges Predictor</h1>', unsafe_allow_html=True)
# st.markdown('🌟 <span class="blue-text">Contact: sxmzhaogb@gmail.com</span>', unsafe_allow_html=True)
# st.markdown('🌟 <span class="green-text">Cite as: GCNCharges ****</span>', unsafe_allow_html=True)

# uploaded_file = st.file_uploader("Please upload your CIF file", type="cif")
# model_option = st.radio("Type", ('MOF', 'COF'))

# if uploaded_file is not None and model_option:
#     file_name = uploaded_file.name.split('.')[0]
#     bytes_data = uploaded_file.getvalue()
#     with open(f'./{file_name}.cif', 'wb') as f:
#         f.write(bytes_data)
#     structure = read(uploaded_file, format='cif')
#     xyz_string_io = StringIO()
#     write(xyz_string_io, structure, format="xyz")
#     xyz_string = xyz_string_io.getvalue()
    
#     formula = structure.get_chemical_formula()
#     st.info(formula, icon="✅")

#     speck_plot(xyz_string, wbox_height="700px", wbox_width="800px",component_h = 700, component_w = 800, scroll = False)
    
#     if st.button(' :sunglasses: :red[Get GCN Charges]', key="predict_button"):
#         prediction = predict_with_model(model_option, f'{file_name}.cif', file_name)
#         if prediction is not None:
#             st.markdown('<span class="green-text">Please download the structure with GCN Charge</span>', unsafe_allow_html=True)
#             st.download_button(label="Download cif file with charges", data=prediction, file_name=f"{file_name}_gcn.cif", mime='text/plain')
#         else:
#             st.error("No data available for download, please check your structure!")

# st.markdown('<span style="color:grey;">Site developed and maintained by Guobin Zhao (Prof.Chung, Yongchul G, Pusan National University)</span>', unsafe_allow_html=True)
# st.markdown("[Molecular Thermodynamics & Advance Processes Laboratory](https://sites.google.com/view/mtap-lab/home)", unsafe_allow_html=True)

import streamlit as st
from predict import predict_with_model
from stmol import *
import py3Dmol
from ase.io import read, write
from io import StringIO
import time
import json
import warnings
import numpy as np
import pymatgen.core as mg
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.cif import CifParser

def n_atom(mof):
    try:
        structure = read(mof)
        struct = AseAtomsAdaptor.get_structure(structure)
    except:
        struct = CifParser(mof, occupancy_tolerance=10)
        struct.get_structures()
    elements = [str(site.specie) for site in structure.sites]
    return len(elements)

st.markdown("""
    <style>
    .big-font {
        font-size:30px !important;
        color: #FF4B4B;
    }
    .blue-text {
        color: #4F8BF9;
    }
    .green-text {
        color: #49BE25;
    }
    .custom-button {
        background-color: #FF4B4B;
        color: white;
        padding: 10px 24px;
        border: none;
        border-radius: 4px;
        cursor: pointer;
    }
    .custom-button:hover {
        background-color: #FF7878;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<h1 class="big-font">💭 MOF/COF GCN Charges Predictor</h1>', unsafe_allow_html=True)
st.markdown('🌟 <span class="blue-text">Contact: sxmzhaogb@gmail.com</span>', unsafe_allow_html=True)
st.markdown('🌟 <span class="grey-text">Cite as: A Robust Partial Atomic Charge Estimator for Nanoporous Materials using Crystal Graph Convolution Network ****</span>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Please upload your CIF file", type="cif")
model_option = st.radio("Type", ('MOF', 'COF'))

if uploaded_file is not None and model_option:
    file_name = uploaded_file.name.split('.')[0]
    bytes_data = uploaded_file.getvalue()
    with open(f'./{file_name}.cif', 'wb') as f:
        f.write(bytes_data)
    structure = read(uploaded_file, format='cif')
    xyz_string_io = StringIO()
    write(xyz_string_io, structure, format="xyz")
    xyz_string = xyz_string_io.getvalue()
    
    formula = structure.get_chemical_formula()
    st.info(formula, icon="✅")

    speck_plot(xyz_string, wbox_height="700px", wbox_width="800px",component_h = 700, component_w = 800, scroll = False)
    
    if st.button(' :sunglasses: :red[Get GCN Charges]', key="predict_button"):
        n_atoms = n_atom(structure)
        if n_atoms<=300:
            total_time = 3
        elif 300<n_atoms<=500:
            total_time = 6
        elif 500<n_atoms<=1000:
            total_time = 10
        elif 1000<n_atoms<=2500:
            total_time = 30
        elif 2500<n_atoms<=5000:
            total_time = 90
        elif 5000<n_atoms<=8000:
            total_time = 300
        elif 8000<n_atoms<=16000:
            total_time = 900
        elif 16000<n_atoms:
            total_time = 10000

        with st.spinner('Processing...'):
            progress_bar = st.progress(0)
            for i in range(100):
                # Simulate processing time
                time.sleep(total_time / 100)
                progress_bar.progress(i + 1)
        
        prediction = predict_with_model(model_option, f'{file_name}.cif', file_name)
        if prediction is not None:
            st.markdown('<span class="green-text">Please download the structure with GCN Charge</span>', unsafe_allow_html=True)
            st.download_button(label="Download cif file with charges", data=prediction, file_name=f"{file_name}_gcn.cif", mime='text/plain')
        else:
            st.error("No data available for download, please check your structure!")

st.markdown('<span style="color:grey;">Site developed and maintained by Guobin Zhao (Prof.Chung, Yongchul G, Pusan National University)</span>', unsafe_allow_html=True)
st.markdown("[Molecular Thermodynamics & Advance Processes Laboratory](https://sites.google.com/view/mtap-lab/home)", unsafe_allow_html=True)
