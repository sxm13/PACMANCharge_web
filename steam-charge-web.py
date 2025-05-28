import os
import torch

torch.classes.__path__ = []

import time
import py3Dmol
from stmol import *
import streamlit as st
from io import StringIO
from ase.io import read, write
from predict import predict_with_model

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
        PACMAN: A <span class="blue">P</span>artial <span class="blue">A</span>tomic <span class="blue">C</span>harge Predicter for Porous <span class="blue">Ma</span>terials based on Graph Convolutional Neural <span class="blue">N</span>etworks
    </p>
    """, unsafe_allow_html=True)
st.subheader('', divider='rainbow')

uploaded_file = st.file_uploader("Please upload your CIF file", type="cif")
st.markdown("""
            <style>
            .big-font {
            font-size:14px !important;
            color: grey;
            }
                </style>
            <p class="big-font">Note: Your CIF file must can be read by ASE or Pymatgen (check is there a "data_" word in you CIF flie).</p>
            """, unsafe_allow_html=True)

charge_option = st.radio(
                        "Charge Type",
                        ["DDEC6", "Bader", "CM5", "REPEAT"],
                        index=0,
                        key="charge model"
                        )
atom_type_option = st.radio(
                "Atom Type",
                [True, False],
                index=0,
                key="keep atom type"
                )

st.markdown("""
            <style>
            .big-font {
            font-size:14px !important;
            color: grey;
            }
                </style>
            <p class="big-font">Keep the same partial atomic charge for the same atom types (based on the similarity of partial atomic charges up to 3 decimal places).</p>
            """, unsafe_allow_html=True)

neutral_option = st.radio(
                "Neutral",
                [True, False],
                index=0,
                key="keep zero"
                )
st.markdown("""
            <style>
            .big-font {
            font-size:14px !important;
            color: grey;
            }
                </style>
            <p class="big-font">Keep the net charge is zero. We use "mean" method to neuralize the system where the excess charges are equally distributed across all atoms.</p>
            """, unsafe_allow_html=True)

connect_option = st.radio(
                "Keep Connect",
                [True, False],
                index=1,
                key="keep bond"
                )
st.markdown("""
            <style>
            .big-font {
            font-size:14px !important;
            color: grey;
            }
                </style>
            <p class="big-font">Retain the atomic and connection information (such as _atom_site_adp_type, bond) for the structure.</p>
            """, unsafe_allow_html=True)

digits = st.number_input("Digits", min_value=1, value=6, max_value=15)
st.markdown("""
            <style>
            .big-font {
            font-size:14px !important;
            color: grey;
            }
                </style>
            <p class="big-font">Note: models are trained on 6-digit data.</p>
            """, unsafe_allow_html=True)


if uploaded_file is not None:
    file_name = uploaded_file.name.split('.')[0]
    bytes_data = uploaded_file.getvalue()
    
    try:
        temp_file_path = f'./{file_name}.cif'
        with open(temp_file_path, 'wb') as f:
            f.write(bytes_data)
        
        structure = read(temp_file_path, format='cif')
        
        xyz_string_io = StringIO()
        write(xyz_string_io, structure, format="xyz")
        xyz_string = xyz_string_io.getvalue()
        
        xyzview = py3Dmol.view(width=800, height=500)
        xyzview.addModel(xyz_string, "xyz")
        xyzview.setStyle({'stick': {}})
        xyzview.zoomTo()
        showmol(xyzview, height=500, width=800)



        #  _xyz = st.text_area(
        #             label = "Enter xyz coordinates below ⬇️",
        #             value = xyz_string, height  = 200)
        # st.success(_xyz.splitlines()[1],icon="✅")
        # res = speck_plot(_xyz,wbox_height="500px",wbox_width="500px")


        formula = structure.get_chemical_formula()
        n_atoms = len(structure)
        st.markdown(f"**Chemical Formula:** {formula}")
        st.markdown(f"**Number of Atoms:** {n_atoms}")
        
    except Exception as e:
        st.error(f"Error processing CIF file: {e}")

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
        
        prediction, atom_type_count, net_charge = predict_with_model(charge_option, f'{file_name}.cif', file_name, digits, atom_type_option, neutral_option, connect_option)
        if prediction is not None:
            if atom_type_option:
                st.write("Atom: number of type")
                st.write(atom_type_count)
            if not neutral_option:
                st.write(f'Net Charge: {net_charge}')
            st.markdown('<span class="green-text">Please download the structure with PACMAN Charge</span>', unsafe_allow_html=True)
            st.download_button(label="Download cif file with charges", data=prediction, file_name=f"{file_name}_pacman.cif", mime='text/plain')
        else:
            st.error("No data available for download, please check your structure!")

st.markdown('* [Source code in github](https://github.com/mtap-research/PACMAN-charge)', unsafe_allow_html=True)            
st.markdown('* <span class="grey-text">Cite as: [Zhao, Guobin and Chung, Yongchul. PACMAN: A Robust Partial Atomic Charge Predicter for Nanoporous Materials based on Crystal Graph Convolution Network. 2024](https://pubs.acs.org/doi/10.1021/acs.jctc.4c00434). </span>', unsafe_allow_html=True)
st.markdown('* <span class="blue-text">Email: sxmzhaogb@gmail.com</span>', unsafe_allow_html=True)
st.markdown("* [Molecular Thermodynamics & Advance Processes Laboratory](https://sites.google.com/view/mtap-lab/home?authuser=0)")

st.markdown("""
            <style>
            .big-font {
            font-size:14px !important;
            color: grey;
            }
                </style>
            <p class="big-font">Version 1.3.</p>
            """, unsafe_allow_html=True)
