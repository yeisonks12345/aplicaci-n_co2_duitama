import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
# Carga del modelo desde un archivo pickle
input_dfd = pd.read_excel('Calidad_del_Aire_Municipio_de_Duitama_20240623.xlsx')
model =pickle.load(open('co2_predic.pkl','rb'))


# Título de la app
st.title("App para estimar las emisiones de CO2 en Duitama Boyacá mediante un modelo de Machine learning XGBoost Regressor")
st.write('A partir de las mediciones generados por sensores ubicados en Duitama Boyacá relacionados con la calidad del aire, se estima la emisión de CO2 en la ciudad, los sensores fueron desarrollados por el Centro internacional de Fisica de la Uiversidad Nacional')
st.write('Las variables que se usaron para estimar la emisón de co2 son: Temperatura, humedad relativa, pm2.5, pm5, pm10 (micrometros)')
st.subheader("Descripción del aplicativo")
st.write("Esta aplicación fue desarrollada usando librerias de python para ciencia de datos como pandas, matplotlib, sklearn y el framework Streamlit, en la parte izquierda encontraran 5 barras deslizantes que representan las variables que predicen las emisión de co2, al mover las barras se evidencia como cambia el valor del potencial de emisión de CO2.")
st.subheader("Modelo Xg boost for regression")
st.write('La base de los modelos xg boost son los arboles de decisión, han tomado relevante importancia en el machine learning por su rendimiento frente a otros modelos, fue desarrollado por Chen Tianqui Carlos Guestrin en 2026, se caracteriza por usar d eforma reitara el aumento del gradiente (Ikbal & Abbas, 2023). ')
# Crear sliders en el sidebar para las 4 variables
st.sidebar.header("Ajuste de Variables")
var1 = st.sidebar.slider("pm10", min_value=0.0, max_value=1792.0, value=900.0)
var2 = st.sidebar.slider("pm2_5", min_value=0.0, max_value=3584.0,  value=1500.0)
var3 = st.sidebar.slider("pm5", min_value=0.0, max_value=3072.0, value=1500.0)
var4 = st.sidebar.slider("Humedad relativa", min_value=0.0, max_value=75669.0, value=50000.0)
var5 = st.sidebar.slider("Temperatura", min_value=0.0, max_value=30.6, value=15.0)

# Convertir las variables a un array numpy para hacer la predicción
input_data = np.array([[var1, var2, var3, var4,var5]])

# Hacer la predicción
prediction = model.predict(input_data)

# Mostrar la predicción
st.subheader("A partir de los variables seleccionadas el potencial de CO2 es:")
highlight_css = """
<div style="
    border: 2px solid #4CAF50; 
    padding: 10px; 
    border-radius: 10px; 
    background-color: #f9f9f9; 
    text-align: center; 
    font-size: 24px; 
    font-weight: bold; 
    color: #4CAF50;
">
    {0}
</div>
"""

# Mostrar el dato enmarcado y resaltado
st.markdown(highlight_css.format(prediction[0]), unsafe_allow_html=True)
#################################################

st.subheader("Métrica F socore")
st.write("En la siguiente imagen se relaciona la importancia de cada variable en el modelo, para determinarlo se combinan la precisión y la sensibilidad. La varaible con mayor F score implica alta importancia en la predicción del modelo.")
data = pd.read_excel('Calidad_del_Aire_Municipio_de_Duitama_20240623.xlsx')

features = data.iloc[:, 1:-1]
target = data.iloc[:, -1]


Xtrain,Xtest,Ytrain,Ytest=train_test_split(features,target,test_size=0.2,random_state=2)
import xgboost as xgb
model = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                              learning_rate=0.05, max_depth=4, 
                              min_child_weight=1.7817, n_estimators=2200,
                              reg_alpha=0.4640, reg_lambda=0.8571,
                               subsample=0.5213, silent=1,
                              random_state =42, nthread = -1)
model.fit(Xtrain,Ytrain)


Ypred = model.predict(Xtest)

fig, ax = plt.subplots()
xgb.plot_importance(model,ax=ax)
st.pyplot(fig)
