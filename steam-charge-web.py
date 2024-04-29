import streamlit as st
from predict import predict_with_model
from stmol import *
import py3Dmol
from ase.io import read, write
from io import StringIO
import time

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

st.markdown("""
    <style>
    .title-font {
        font-size:24px;  
        font-weight:bold; 
    }
    .blue {
        color: blue;  
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("""
    <p class="title-font">
        PACMAN: <span class="blue">P</span>artial <span class="blue">A</span>tomic <span class="blue">C</span>harges Predicter for Porous <span class="blue">Ma</span>terials based on Graph Convolutional Neural <span class="blue">N</span>etwork
    </p>
    """, unsafe_allow_html=True)
st.subheader('', divider='rainbow')

uploaded_file = st.file_uploader("Please upload your CIF file", type="cif")

model_option = st.radio("Material Type", ('MOF', 'COF'))
if model_option == 'COF':
    charge_option = st.radio("Charge Type", ('DDEC6',))
else:
    charge_option = st.radio("Charge Type", ('DDEC6', 'Bader', 'CM5'))

st.markdown("""
            <style>
            .big-font {
            font-size:14px !important;
            color: black;
            }
                </style>
            <p class="big-font">Note: just DDEC6 can be used for COF .</p>
            """, unsafe_allow_html=True)

digits = st.number_input("Digits", min_value=1, value=6)
st.markdown("""
            <style>
            .big-font {
            font-size:14px !important;
            color: black;
            }
                </style>
            <p class="big-font">Note: models are trained on 6-digit data.</p>
            """, unsafe_allow_html=True)

atom_type_option = st.radio("Atom Type", ('Yes', 'No'))

neutral_option = st.radio("Neutral", ('Yes', 'No'))


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
    st.info(f"Formula: {formula}", icon="✅")

    speck_plot(xyz_string, wbox_height="700px", wbox_width="800px",component_h = 700, component_w = 800, scroll = False)
    
    n_atoms = len(structure)
    st.markdown(f'Number of atoms: **{n_atoms}**')

    if st.button(':rainbow[Get PACMAN Charge]', key="predict_button"):

        if n_atoms <= 300:
            total_time = 15
        elif 300 < n_atoms <= 500:
            total_time = 30
        elif 500 < n_atoms <= 1000:
            total_time = 45
        elif 1000 < n_atoms <= 2500:
            total_time = 135
        elif 2500 < n_atoms <= 5000:
            total_time = 450
        elif 5000 < n_atoms <= 8000:
            total_time = 1500
        elif 8000 < n_atoms <= 16000:
            total_time = 4500
        else:
            total_time = 10000  # For more than 16000 atoms
        
        st.markdown(f'Estimated processing time: **{total_time} seconds**')

        with st.spinner('Processing...'):
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(total_time / 100)
                progress_bar.progress(i + 1)
        
        prediction, atom_type_count, net_charge = predict_with_model(model_option, charge_option, f'{file_name}.cif', file_name, digits, atom_type_option, neutral_option)
        if prediction is not None:
            if atom_type_option == 'Yes':
                st.write("Atom: number of type")
                st.write(atom_type_count)
            if neutral_option == 'No':
                st.write(f'Net Charge: {net_charge}')
            st.markdown('<span class="green-text">Please download the structure with PACMAN Charge</span>', unsafe_allow_html=True)
            st.download_button(label="Download cif file with charges", data=prediction, file_name=f"{file_name}_pacman.cif", mime='text/plain')
        else:
            st.error("No data available for download, please check your structure!")

st.markdown('* [Source code in github](https://github.com/sxm13/PACMAN)', unsafe_allow_html=True)            
st.markdown('* <span class="grey-text">Cite as: Zhao, Guobin and Chung, Yongchul. A Robust Partial Atomic Charge Estimator for Nanoporous Materials using Crystal Graph Convolution Network. Journal of Chemical Theory and Computation. 2024. </span>', unsafe_allow_html=True)
st.markdown('* <span class="blue-text">Email: sxmzhaogb@gmail.com</span>', unsafe_allow_html=True)
st.markdown("* [Molecular Thermodynamics & Advance Processes Laboratory](https://sites.google.com/view/mtap-lab/home?authuser=0)")
