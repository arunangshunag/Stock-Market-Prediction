import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt

model = load_model(r'C:\Users\Arunangshu_PC\Desktop\stock market prediciton\Stock Predictions Model.keras')

st.header('Stock Market Predictor')

stock =st.text_input('Enter Stock Symnbol', 'GOOG')
start = '2015-01-01'
end = '2024-2-20'

data = yf.download(stock, start ,end)

st.subheader('Stock Data From 2015 - 2024')
st.write(data)

st.subheader('Stock Closing price Line Chart')
data.drop(columns = ['Open','High','Low','Adj Close','Volume'], inplace=True)
st.line_chart(data)

data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

st.subheader('Price vs MA100')
st.write('100 MA IS THE AVG OF THE PREVIOUS 100 DAYS CLOSING VALUE AND PLOTTED ')
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8,6))
plt.plot(ma_100_days, 'b')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig2)

st.subheader('Price vs MA100 vs MA200')
st.write('IF 100 DAYS MA CROSSES ABOVE 200 DAYS MA THEN UPTREND OR IF 100 DAYS MA CROSSES BELOW 200 DAYS MA THEN DOWNTREND')
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8,6))
plt.plot(ma_100_days, 'r',label= 'MA100')
plt.plot(ma_200_days, 'b',label= 'MA200')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig3)

x = []
y = []
for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])

x,y = np.array(x), np.array(y)
x = np.reshape(x, (x.shape[0],x.shape[1],1))

pred = model.predict(x)
scale = 1/scaler.scale_
pred = pred * scale
y = y * scale

st.subheader('Original Price vs Predicted Price')
preds = pred.reshape(-1,1)
#ys = scaler.inverse_transform(y.reshape(-1,1))
preds = pd.DataFrame(preds, columns=['Predicted Price'])
ys = pd.DataFrame(y, columns=['Original Price'])
chart_data = pd.concat((preds, ys), axis=1)
st.write(chart_data)

fig4 = plt.figure(figsize=(8,6))
plt.plot(pred, 'r', label='Original Price')
plt.plot(y, 'y', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()
st.pyplot(fig4)

st.subheader('FUTURE PRICE PREDICTION OF NEXT 5 DAYS')
future_days = 5
predicted_prices = []

# Take the last 100 days from the training data as a starting point for prediction
last_100_days = data_test_scale[-100:].reshape(1, -1)

for i in range(future_days):
    # Reshape the last 100 days data for prediction
    input_data = last_100_days.reshape((1, 100, 1))
    # Predict the next day's price
    prediction = model.predict(input_data)
    # Append the prediction to the list
    predicted_prices.append(prediction[0,0])
    # Update the last_100_days array for the next iteration
    last_100_days = np.append(last_100_days[:,1:], prediction, axis=1)

predicted_prices = np.array(predicted_prices)
predicted_prices = scaler.inverse_transform(predicted_prices.reshape(-1,1))
st.write(predicted_prices)
st.line_chart(predicted_prices)
