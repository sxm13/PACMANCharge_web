import streamlit as st
from predict import predict_with_model
import py3Dmol

def show_structure(cif_file):
    viewer = py3Dmol.view(width=800, height=400)
    with open(cif_file, 'r') as file:
        cif_data = file.read()
    viewer.addModel(cif_data, 'cif')
    viewer.setStyle({'stick': {}})
    viewer.zoomTo()
    return viewer.show()

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
st.markdown('🌟 <span class="green-text">Cite as: GCNCharges ****</span>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Please upload your CIF file", type="cif")
model_option = st.radio("Type", ('MOF', 'COF'))

if uploaded_file is not None and model_option:
    file_name = uploaded_file.name.split('.')[0]
    bytes_data = uploaded_file.getvalue()
    with open(f'./{file_name}.cif', 'wb') as f:
        f.write(bytes_data)
    st.markdown("### Your Structure")
    st.markdown(show_structure(f'./{file_name}.cif'), unsafe_allow_html=True)
    if st.button('Get GCN Charges', key="predict_button"):
        prediction = predict_with_model(model_option, f'{file_name}.cif', file_name)
        if prediction is not None:
            st.markdown('<span class="green-text">Please download the structure with GCN Charge</span>', unsafe_allow_html=True)
            st.download_button(label="Download cif file with charges", data=prediction, file_name=f"{file_name}_gcn.cif", mime='text/plain')
        else:
            st.error("No data available for download, please check your structure!")

st.markdown('<span style="color:grey;">Site developed and maintained by Guobin Zhao (Prof.Chung, Yongchul G, Pusan National University)</span>', unsafe_allow_html=True)
st.markdown("[Molecular Thermodynamics & Advance Processes Laboratory](https://sites.google.com/view/mtap-lab/home)", unsafe_allow_html=True)
