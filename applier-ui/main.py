import os
import pandas as pd
import streamlit as st
import requests
from requests.auth import HTTPBasicAuth
import dotenv

from src.dates import get_previous_month_name
from src.models import load_models, get_model_name, load_model_to_dict
from src.constants import IRON, CABLING, TIMBER, CEMENT, READY_MIXED_CONCRETE, PRODUCTS
from src.discretizer import discretize, reverse

# Load env variables (default filename .env)
dotenv.load_dotenv()

# Path to downloaded models
MODELS_PATH = 'data/models'

# Set page config
st.set_page_config(layout="wide", page_title="Material Prices Prediction")

models = load_models(MODELS_PATH)

model_file = st.selectbox(label="Model", options=models, format_func=get_model_name)

col1, col2, col3 = st.columns(3, gap='medium')

with col1:
    st.subheader(f"Prices: {get_previous_month_name(1)}")
    iron1 = st.number_input(label="Iron", key=f"1-iron", step=100)
    cabling1 = st.number_input(label="Cabling", key=f"1-cabling", step=100)
    timber1 = st.number_input(label="Timber", key="1-timber", step=100)
    cement1 = st.number_input(label="Cement", key="1-cement", step=1)
    ready_mixed_concrete1 = st.number_input(label="Ready Mixed Concrete", key="1-ready-mixed-concrete", step=10)

with col2:
    st.subheader(f"Prices: {get_previous_month_name(2)}")
    iron2 = st.number_input(label="Iron", key=f"2-iron", step=100)
    cabling2 = st.number_input(label="Cabling", key=f"2-cabling", step=100)
    timber2 = st.number_input(label="Timber", key="2-timber", step=100)
    cement2 = st.number_input(label="Cement", key="2-cement", step=1)
    ready_mixed_concrete2 = st.number_input(label="Ready Mixed Concrete", key="2-ready-mixed-concrete", step=10)

with col3:
    st.subheader(f"Prices: {get_previous_month_name(3)}")
    iron3 = st.number_input(label="Iron", key=f"3-iron", step=300)
    cabling3 = st.number_input(label="Cabling", key=f"3-cabling", step=300)
    timber3 = st.number_input(label="Timber", key="3-timber", step=300)
    cement3 = st.number_input(label="Cement", key="3-cement", step=3)
    ready_mixed_concrete3 = st.number_input(label="Ready Mixed Concrete", key="3-ready-mixed-concrete", step=30)

predict = st.button(label="Predict")

if predict:
    model = load_model_to_dict(model_file)
    st.write("Predicting...")
    st.write("Using model: ", model_file)
        
    iron_prices = [iron1, iron2, iron3]
    cabling_prices = [cabling1, cabling2, cabling3]
    timber_prices = [timber1, timber2, timber3]
    cement_prices = [cement1, cement2, cement3]
    ready_mixed_concrete_prices = [ready_mixed_concrete1, ready_mixed_concrete2, ready_mixed_concrete3]
    
    iron_prices = list(map(lambda x: f'IRON_{x}', [discretize(x, IRON['avg'], IRON['stddev']) for x in iron_prices]))
    cabling_prices = list(map(lambda x: f'CABLING_{x}', [discretize(x, CABLING['avg'], CABLING['stddev']) for x in cabling_prices]))
    timber_prices = list(map(lambda x: f'TIMBER_{x}', [discretize(x, TIMBER['avg'], TIMBER['stddev']) for x in timber_prices]))
    cement_prices = list(map(lambda x: f'CEMENT_{x}', [discretize(x, CEMENT['avg'], CEMENT['stddev']) for x in cement_prices]))
    ready_mixed_concrete_prices = list(map(lambda x: f'READY_MIXED_CONCRETE_{x}', [discretize(x, READY_MIXED_CONCRETE['avg'], READY_MIXED_CONCRETE['stddev']) for x in ready_mixed_concrete_prices]))
    
    iron_prices = [f'{y}:{x}' for (x, y) in zip(iron_prices, ['PM', 'PM2', 'PM3'])]
    cabling_prices = [f'{y}:{x}' for (x, y) in zip(cabling_prices, ['PM', 'PM2', 'PM3'])]
    timber_prices = [f'{y}:{x}' for (x, y) in zip(timber_prices, ['PM', 'PM2', 'PM3'])]
    cement_prices = [f'{y}:{x}' for (x, y) in zip(cement_prices, ['PM', 'PM2', 'PM3'])]
    ready_mixed_concrete_prices = [f'{y}:{x}' for (x, y) in zip(ready_mixed_concrete_prices, ['PM', 'PM2', 'PM3'])]
           
    instance = iron_prices + cabling_prices + timber_prices + cement_prices + ready_mixed_concrete_prices
    instance = ",".join(instance)
    
    payload = {
        'instance': instance,
        'model': model
    }
    
    applier_url = os.environ['APPLIER_URL']
    username = os.environ['SKEYEPREDICT_USERNAME']
    password = os.environ['SKEYEPREDICT_PASSWORD']

    headers = {'Content-type': 'application/json'}

    basic = HTTPBasicAuth(username=username, password=password)
    response = requests.post(applier_url, json=payload, auth=basic, headers=headers)

    result = response.json()
    
    if result['errors'] != {}:
        st.error(f"API Error: {result['errors']}")
    else:
        st.subheader("Predictions: ")
        prediction_df = pd.DataFrame(data=result['predictions'])

        for _, row in prediction_df.iterrows():
            annotation = row['annotation']
            product = annotation.split(':')[0]
            price_level = annotation.split(':')[1]
            confidence = row['confidence']
        
            stats_map = PRODUCTS[product]
            min_price, max_price = reverse(level=price_level, avg=stats_map['avg'], stddev=stats_map['stddev'])
            
            st.write(f"Product {product} will have a price level of {price_level} This month, Between {min_price:.2f} and {max_price:.2f}")
            st.info(f"Confidence Level: {confidence * 100:.2f} %")

        st.warning("========================================================================================")                
        st.write(prediction_df)
