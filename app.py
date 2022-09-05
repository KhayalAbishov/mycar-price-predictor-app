"""
Data Scientist: Khayal Abishov      
Created: 05.09.2022
"""

# -*- coding: utf-8 -*-
"""
Data Scientist: Khayal Abishov        

Created: 05.09.2022
"""

import streamlit as st
import pickle
import numpy as np
import pandas as pd

data = pd.read_csv('cleaned_data.csv')
pipe = pickle.load(open('model.pkl', 'rb'))

def predict_car_price(city,brand,model,year,body,color,engine_power,fuel_type,mileage,gearbox,transmission,new):
    query = np.array([city,brand,model,year,body,color,engine_power,fuel_type,mileage,gearbox,transmission,new])
    query = query.reshape(1, 12)
    df = pd.DataFrame(data=query, index=np.arange(len(query)), columns=["City", "Brand", "Model", "Year", "Body", "Color", 
                "Engine_Power", "Fuel_Type", "Mileage", "Gearbox", "Transmission", "New"])
    prediction=pipe.predict(df)
    st.title(f'The predicted price is {prediction[0]} AZN')
    return prediction


def main():
    html_temp = """
    <div style="background-color:black;padding:10px">
    <h2 style="color:white;text-align:center;">Car Price Predictor - ML App</h2> <p style="color:white;text-align:center;">Data have been scraped from Turbo.az</p>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    car_dict = {}
    for x in data['Brand'].unique():
        l = []
        for y in data[data['Brand'] == x]['Model'].unique():
            l.append(y)
        car_dict[x]=l


    brand = st.selectbox("Brand", data['Brand'].sort_values().unique())
    model = st.selectbox("Model", sorted(car_dict[brand]))
    city = st.selectbox('City', data['City'].sort_values().unique())
    year = st.selectbox('Year', data['Year'].sort_values().unique())
    body = st.selectbox('Body', data['Body'].sort_values().unique())
    color = st.selectbox('Color', data['Color'].sort_values().unique())
    engine_power = st.number_input('Engine_Power', min_value=0.0, value=100.0, step=10.0)
    fuel_type = st.selectbox('Fuel_Type', data['Fuel_Type'].sort_values().unique())
    mileage = st.number_input('Mileage', min_value=0.0, max_value=300000.0, value=100000.0, step=10000.0)
    gearbox = st.selectbox('Gearbox', data['Gearbox'].sort_values().unique())
    transmission = st.selectbox('Transmission', data['Transmission'].sort_values().unique())
    new = st.selectbox('New', data['New'].sort_values().unique())


    if st.button('Predict Price'):
        return predict_car_price(city,brand,model,year,body,color,engine_power,fuel_type,mileage,gearbox,transmission,new)


if __name__=='__main__':
    main()
