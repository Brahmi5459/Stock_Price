import streamlit as st
import pickle 
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load the trained model
model = load_model('C:\\Users\\Brahmi\\OneDrive\\Documents\\4-2 sem\\4-2 Project\\model.h5')

# Load the stock data
data = pd.read_csv("C:/Users/Brahmi/OneDrive/Documents/4-2 sem/all_stocks_5yr.csv")

# Use 'close' prices for simplicity (you might want to use more features for a real-world scenario)
features = data[['close']].values

# Scale the features
scaler = MinMaxScaler(feature_range=(0, 1))
features_scaled = scaler.fit_transform(features)

# Define sequence length for time series data
sequence_length = 10

# Streamlit app
st.title('Stock Price Prediction')

company_name = st.text_input("Enter Company Name: ")

if st.button('Predict'):
    if company_name not in data['Name'].unique():
        st.write(f"Company '{company_name}' not found in the dataset.")
    else:
        # Load data for the specified company
        company_data = data[data['Name'] == company_name].copy()
        company_features = company_data[['close']].values

        # Scale the features
        company_features_scaled = scaler.transform(company_features)

        predicted_prices = []

        for i in range(7):  # Predicting next 7 days
            input_sequence = company_features_scaled[-sequence_length:].reshape(1, sequence_length, 1)
            predicted_price_scaled = model.predict(input_sequence)
            predicted_price = scaler.inverse_transform(predicted_price_scaled.reshape(-1, 1))
            predicted_prices.append(predicted_price[0][0])

            # Append the predicted price to the input sequence for the next prediction
            company_features_scaled = np.append(company_features_scaled, predicted_price_scaled)
            company_features_scaled = company_features_scaled[1:]

        # Print predicted prices for the next 7 days
        st.subheader(f"Predicted Close Prices for the next 7 days for {company_name}:")
        for day, price in enumerate(predicted_prices, start=1):
            st.write(f"Day {day}: ${price:.2f}")
