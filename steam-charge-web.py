import streamlit as st
from predict import predict_with_model

st.title('MOF/COF GCN Charges Predicter')
uploaded_file = st.file_uploader("Please upload your CIF file", type="cif")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    with open('./upload.cif', 'wb') as f:
        f.write(bytes_data)
model_option = st.radio("Type", ('MOF', 'COF'))

if uploaded_file is not None and model_option:
    prediction = predict_with_model(model_option, 'upload.cif')
    st.write("predicted result: ")
    st.write(prediction)
    if prediction is not None:
        st.download_button(label="Download cif file with charges", data=prediction, file_name="prediction.cif")
    else:
        st.write("No data available for download")
