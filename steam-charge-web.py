import streamlit as st
from predict import predict_with_model

st.title('MOF/COF GCN Charges Predictor')
st.markdown("Contact: sxmzhaogb@gmail.com")
st.markdown(' :heart_eyes: <span style="color:grey;">Cite as: GCNCharges ****</span>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Please upload your CIF file", type="cif")
if uploaded_file is not None:
    # Extract the file name without extension
    file_name = uploaded_file.name.split('.')[0]

    bytes_data = uploaded_file.getvalue()
    with open(f'./{file_name}.cif', 'wb') as f:
        f.write(bytes_data)

model_option = st.radio("Type", ('MOF', 'COF'))

if uploaded_file is not None and model_option:
    prediction = predict_with_model(model_option, f'{file_name}.cif')
    st.write("predicting")
    if prediction is not None:
        # Use the input file name for the output file
        st.download_button(label="Download cif file with charges", data=prediction, file_name=f"{file_name}_prediction.cif")
    else:
        st.write("No data available for download")

st.markdown('<span style="color:grey;">Site developed and maintained by Guobin Zhao (Pusan National University)</span>', unsafe_allow_html=True)
