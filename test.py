from predict import predict_with_model

result,atom_type_count,net_charge = predict_with_model("DDEC6", "DOTSOV01_clean.cif","DOTSOV01_clean", 10, True, True)
print(result,atom_type_count,net_charge)