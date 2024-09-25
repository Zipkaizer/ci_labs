import pandas as pd 
import numpy as np
import os 
import pickle as pk 
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

model = pk.load(open('/absolute/path/to/ci_labs/lab1/model.pkl', 'rb'))

st.header('Car Price Prediction ML Model')

cars_data = pd.read_csv('/absolute/path/to/ci_labs/lab1/Cardetails.csv')

def get_brand_name(car_name):
    car_name = car_name.split(' ')[0]
    return car_name.strip()

cars_data['name'] = cars_data['name'].apply(get_brand_name)

# User inputs
name = st.selectbox('Select Car Brand', cars_data['name'].unique())
year = st.slider('Car Manufactured Year', 1994, 2024)
km_driven = st.slider('No of kms Driven', 11, 200000)
fuel = st.selectbox('Fuel type', cars_data['fuel'].unique())
seller_type = st.selectbox('Seller type', cars_data['seller_type'].unique())
transmission = st.selectbox('Transmission type', cars_data['transmission'].unique())
owner = st.selectbox('Owner type', cars_data['owner'].unique())
mileage = st.slider('Car Mileage', 10, 40)
engine = st.slider('Engine CC', 700, 5000)
max_power = st.slider('Max Power', 0, 200)
seats = st.slider('No of Seats', 5, 10)

# Prediction and Visualization
if st.button("Predict"):
    # Create input DataFrame for the prediction
    input_data_model = pd.DataFrame(
        [[name, year, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, seats]],
        columns=['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats']
    )
    
    # Apply necessary replacements
    input_data_model['owner'].replace(['First Owner', 'Second Owner', 'Third Owner', 
                                       'Fourth & Above Owner', 'Test Drive Car'],
                                      [1, 2, 3, 4, 5], inplace=True)
    input_data_model['fuel'].replace(['Diesel', 'Petrol', 'LPG', 'CNG'], [1, 2, 3, 4], inplace=True)
    input_data_model['seller_type'].replace(['Individual', 'Dealer', 'Trustmark Dealer'], [1, 2, 3], inplace=True)
    input_data_model['transmission'].replace(['Manual', 'Automatic'], [1, 2], inplace=True)
    input_data_model['name'].replace(
        ['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault',
         'Mahindra', 'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz',
         'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan', 'Lexus',
         'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Fiat', 'Force',
         'Ambassador', 'Ashok', 'Isuzu', 'Opel'],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
         21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31], inplace=True)
    
    # Perform prediction
    predicted_price = model.predict(input_data_model)[0]
    st.write(f'Predicted Price: {predicted_price}')
    
    # Visualization: Price vs Mileage for the selected car brand
    filtered_data = cars_data[cars_data['name'] == name]
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=filtered_data['mileage'], y=filtered_data['selling_price'], color='blue')
    sns.lineplot(x=[mileage], y=[predicted_price], marker='o', color='red', label='Your Prediction')
    plt.title(f'Price vs Mileage for {name}')
    plt.xlabel('Mileage (km/l)')
    plt.ylabel('Selling Price')
    st.pyplot(plt)

