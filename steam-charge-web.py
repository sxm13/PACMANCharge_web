import streamlit as st
from predict import predict_with_model

st.title('MOF/COF GCN Charges Predicter')                            
st.markdown("Contact: sxmzhaogb@gmail.com")
st.markdown(' :heart_eyes: <span style="color:grey;">Cite as: GCNCharges ****</span>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Please upload your CIF file", type="cif")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    with open('./upload.cif', 'wb') as f:
        f.write(bytes_data)
model_option = st.radio("Type", ('MOF', 'COF'))

if uploaded_file is not None and model_option:
    prediction = predict_with_model(model_option, 'upload.cif')
    st.write("predicting")
    if prediction is not None:
        st.download_button(label="Download cif file with charges", data=prediction, file_name="prediction.cif")
    else:
        st.write("No data available for download")

st.markdown('<span style="color:grey;">Site developed and maintained by Guobin Zhao (Pusan National University)</span>', unsafe_allow_html=True)
