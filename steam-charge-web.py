import streamlit as st
from predict import predict_with_model

st.title('💭 MOF/COF GCN Charges Predictor')
st.markdown(' :feelsgood: <span style="color:black;">Contact: sxmzhaogb@gmail.com</span>', unsafe_allow_html=True)
st.markdown(' :heart_eyes: <span style="color:grey;">Cite as: GCNCharges ****</span>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Please upload your CIF file", type="cif")
if uploaded_file is not None:
    file_name = uploaded_file.name.split('.')[0]
    bytes_data = uploaded_file.getvalue()
    with open(f'./{file_name}.cif', 'wb') as f:
        f.write(bytes_data)
model_option = st.radio("Type", ('MOF', 'COF'))
if uploaded_file is not None and model_option:
    prediction = predict_with_model(model_option, f'{file_name}.cif',file_name)
    st.write("please download structure with GCN Charge")
    if prediction is not None:
        st.download_button(label="Download cif file with charges", data=prediction, file_name=f"{file_name}_gcn.cif")
    else:
        st.write("No data available for download, please check your structure!")

st.markdown('<span style="color:grey;">Site developed and maintained by Guobin Zhao (Prof.Chung, Yongchul G, Pusan National University)</span>', unsafe_allow_html=True)
st.markdown("[Molecular Thermodynamics & Advance Processes Laboratory](https://sites.google.com/view/mtap-lab/home)")
