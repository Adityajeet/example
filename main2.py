import numpy as np
import pandas as pd
import pickle
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("diabetes_prediction_dataset.csv")
loaded_model = pickle.load(open('trained_model2.sav', 'rb'))


@st.cache_data(show_spinner="Predicting...")
def diabetes_prediction(input_data):
    
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return '**You may not have diabetes.**'
    elif (prediction[0]==1):
      return '**You may have diabetes**'

if __name__ == "__main__":
    st.title("Diabetes classifier")

    with st.expander("See full data table"):
        st.write(df)
    
    st.divider()
   

    

    st.subheader("Select the values for prediction")

    col1,col2 = st.columns(2)



    with col1:
        gender=st.selectbox('**Gender** [0-Female | 1-Male | 2-Other]',(0,1,2))
        age = st.number_input("**Age**", min_value=0.0, max_value=80.0, value=0.0, step=0.02)
        hypertension=st.selectbox("**Hypertension** [0-No|1-Yes]",(0,1))
        heart_disease=st.selectbox("**Heart disease** [0-No|1-Yes]",(0,1))
    with col2:
        HbA1c_level= st.number_input("**hbA1c Level**", min_value=3.5, max_value=9.0, value=3.5, step=0.1)
        bmi = st.number_input("**BMI**", min_value=10.0, max_value=96.0, value=10.01, step=0.01)
        blood_glucose_level= st.number_input("**Blood Glucose Level**", min_value=80, max_value=300, value=140, step=1)
        smoking_history=st.selectbox("**Smoking History** [0 - No-info | 1 - Current | 2 - ever | 3 - former | 4 - never]",(0,1,2,3,4))

    diabetes=""

    pred_button=st.button('Diabetes Test Result',type="primary")

    if pred_button:
       diabetes=diabetes_prediction([gender,age,hypertension,heart_disease,smoking_history,bmi,HbA1c_level,blood_glucose_level])
       st.success(diabetes)

    
    