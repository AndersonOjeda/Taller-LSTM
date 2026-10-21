"""Aplicación: python -m streamlit run app.py"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','1')
import json
import pandas as pd
import numpy as np
import streamlit as st
from tensorflow import keras
from pipeline import ROOT, inference, validate_input, features

st.set_page_config(page_title='Energia · Predicción LSTM', page_icon='⚡',layout='wide')
st.title('⚡ Demanda eléctrica de la siguiente hora')
st.caption('Taller de Inteligencia Artificial · Anderson Ojeda · UCC Pasto')

@st.cache_resource
def load():
    config=json.loads((ROOT/'models/config.json').read_text(encoding='utf-8'))
    model=keras.models.load_model(ROOT/'models/mejor_modelo.keras',compile=False)
    return model,config

try:
    model,config=load()
except (OSError,ValueError) as exc:
    st.error('Faltan los artefactos del entrenamiento o son incompatibles. Ejecuta python train.py.'); st.stop()
w=config['window']
st.info(f'Ingresa {w} horas consecutivas de datos históricos. La aplicación estimará la demanda de la hora inmediatamente posterior, en MW.')
with st.expander('Cómo usar la aplicación'):
    st.write('Carga un CSV o edita el ejemplo. Cada fila representa una hora real. La demanda histórica también es una entrada. festivo: 0 = no; 1 = sí. Las variables del calendario se calculan a partir de las fechas.')
    st.write('El ejemplo procede del dataset del taller. Cambiar valores crea un escenario hipotético; la predicción no prueba que ese cambio cause un efecto real.')
example=pd.read_csv(ROOT/'data/ejemplo_entrada.csv')
mode=st.radio('Datos de entrada',['Editar ejemplo','Cargar CSV'],horizontal=True)
st.download_button('Descargar ejemplo CSV',example.to_csv(index=False).encode('utf-8'),'ejemplo_entrada.csv','text/csv')
if mode=='Cargar CSV':
    uploaded=st.file_uploader('CSV con el histórico completo de la ventana',type=['csv'])
    if uploaded is None: st.stop()
    try: data=pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f'No se pudo leer el archivo: {exc}'); st.stop()
else: data=example
edited=st.data_editor(data,hide_index=True,num_rows='fixed',height=420)
if st.button('Predecir siguiente hora',type='primary'):
    try:
        p,t=inference(edited,model,config)
        a,b=st.columns(2); a.metric('Demanda estimada',f'{p:,.2f} MW'); b.metric('Hora de la predicción',str(t))
        d=validate_input(edited,w)
        chart=pd.DataFrame({'Histórico (MW)':d.demanda_mw})
        chart.loc[t,'Pronóstico (MW)']=p
        st.line_chart(chart)
        scaled=(features(d)-np.array(config['scaler']['mean']))/np.array(config['scaler']['std'])
        if (np.abs(scaled)>5).any(): st.warning('Algunas entradas están muy alejadas de los valores habituales de entrenamiento; interpreta el pronóstico con cautela.')
        st.download_button('Descargar predicción',pd.DataFrame({'timestamp':[t],'demanda_predicha_mw':[p]}).to_csv(index=False),'prediccion.csv','text/csv')
    except (ValueError,KeyError,TypeError) as exc: st.error(str(exc))
with st.expander('Modelo y resultados del taller'):
    st.write(f'Modelo seleccionado por validación: {config["model"]}; ventana: {w} horas.')
    results=ROOT/'results/resultados_modelos.csv'
    if results.exists(): st.dataframe(pd.read_csv(results)[['Modelo','Ventana','MAE','MSE','RMSE','MAPE','R2','Epocas','Seleccionado']],hide_index=True)
    st.caption('Estimación académica a una hora. Las métricas corresponden a ventanas completas y etiquetas verificables del dataset, sin validación en una red eléctrica real.')
