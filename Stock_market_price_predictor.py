import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Streamlit app title
st.title("Stock Market Price Prediction")

# Input for stock symbol
a = st.text_input("Enter the Stock Symbol (Add .NS for Indian stocks):")
if a:
    st.write(f"Fetching data for: {a.upper()}")
    
    # Fetch stock data
ticker = yf.Ticker(a.upper())
df = ticker.history(period="10y")

if not df.empty:
    # Display raw data
    st.subheader("Historical Data")
    st.dataframe(df)

    # Plot closing price history
    st.subheader("Closing Price History")
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(df['Close'])
    ax.set_title("Closing Price History", fontsize=22)
    ax.set_xlabel("Date", fontsize=18)
    ax.set_ylabel("Stock Price", fontsize=18)
    st.pyplot(fig)

    # Prepare data for training
    df_close = df[['Close']]
    df_close_array = df_close.values
    training_data_len = round((len(df_close_array) * 0.8))

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df_close_array)

    train_data = scaled_data[0:training_data_len, :]

    x_train = []
    y_train = []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Define LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    # Compile and train the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    st.write("Training the model... (This might take a while)")
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    # Prepare test data
    test_data = scaled_data[training_data_len - 60:, :]
    x_test = []
    y_test = df_close_array[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Calculate RMSE
    rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
    st.write(f"Root Mean Square Error (RMSE): {rmse}")

    # Plot predictions
    train = df_close[:training_data_len]
    valid = df_close[training_data_len:]
    valid['Predictions'] = predictions

    st.subheader("Model Predictions")
    fig2, ax2 = plt.subplots(figsize=(16, 8))
    ax2.plot(train['Close'], label='Train Data')
    ax2.plot(valid['Close'], label='Test Data')
    ax2.plot(valid['Predictions'], label='Predictions')
    ax2.set_title('Model Predictions', fontsize=22)
    ax2.set_xlabel('Date', fontsize=18)
    ax2.set_ylabel('Close Price', fontsize=18)
    ax2.legend()
    st.pyplot(fig2)

    # Future price prediction
    last_60_days = df_close[-60:].values
    last_60_days_scaled = scaler.transform(last_60_days)
    X_test = []
    X_test.append(last_60_days_scaled)
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    pred_price = model.predict(X_test)
    pred_price = scaler.inverse_transform(pred_price)

    st.subheader("Future Price Prediction")
    st.write(f"Predicted Price for the next day: ${pred_price[0][0]:.2f}")
else:
    st.write("Could not fetch data. Please check the stock symbol.")
