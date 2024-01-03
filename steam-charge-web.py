import streamlit as st
from predict import predict_with_model

st.title('MOF/COF GCN Charges Predicter')
uploaded_file = st.file_uploader("Please upload your CIF file", type="cif")
model_option = st.radio("Type", ('MOF', 'COF'))

if uploaded_file is not None and model_option:
    prediction = predict_with_model(model_option, uploaded_file)
    st.write("predicted result: ")
    st.write(prediction)
    st.download_button(label="Download cif file with charges", data=prediction, file_name="prediction.cif", mime="text/cif")
