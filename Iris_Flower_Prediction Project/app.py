import streamlit as st
import numpy as np
import pickle

with open('iris.pkl','rb') as f:
        model = pickle.load(f)
st.title("Iris Flower Species Prediction")

sepal_lenght = st.slider("Sepal Length", 0.0, 10.0, 5.0)
sepal_width = st.slider("Sepal Width", 0.0, 10.0, 3.0)
petal_lenght = st.slider("Petal Length", 0.0, 10.0, 4.0)
petal_width = st.slider("Petal Width", 0.0, 10.0, 1.0)
input_data = np.array([[sepal_lenght, sepal_width, petal_lenght, petal_width]])

if st.button("Predict"):
        input_data=np.array([sepal_lenght, sepal_width, petal_lenght, petal_width]).reshape(1,-1)
        prediction = model.predict(input_data)
        species =['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
        st.success(f'The predicted species is: {species[prediction[0]]}')
   