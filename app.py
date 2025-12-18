import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
st.header(":blue test_result")
st.subheader("This app will predict your ear test_result giving some features")
#Loading the dataset
#hear = pd.read_csv('hearing_test.csv')
#hear.dataframe(hear)
left_column,right_column= st.columns(2)
age = left_column.number_input('age', min_value = 18.000000, max_value = 90.000000)
physical_score = right_column.number_input('physical_score', min_value = -0.000000, max_value = 50.00000)
#loading the model
with open('knn_reg.sav', 'rb') as m:
    model = pickle.load(m)

#input as dataframes
input_features = ({'age': age, 'physical_score' : physical_score})         

input_df = pd.DataFrame(input_features, index = [0])

predicted_result = model.predict(input_df)
#printing prediction
if st.button('Show predicted_result'):
    st.subheader(f":blue The predicted_result is {predicted_result[0]}.")
