import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from keras.model import load_model
import streamlit as st

st.title("Stock Trend Prediction")

user_input = st.text_input('Enter Stock Ticker','GOOG')
df1 = yf.download(user_input, start = '2013-01-01', end='2023-06-30')
df1.head()

st.subheader("Data from 2013-2023")
st.write(df1.describe())

st.subheader("Closing Price vs Time Chart with 100 MA")
ma100 = df1.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df1.Close)
st.pyplot(fig)

st.subheader("Closing Price vs Time Chart with 100 & 200 MA")
ma100 = df1.Close.rolling(100).mean()
ma200 = df1.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df1.Close)
st.pyplot(fig)

data_training = pd.DataFrame(df1['Close'][0:int(len(df1)*0.70)])
data_testing = pd.DataFrame(df1['Close'][int(len(df1)*0.70):int(len(df1))])


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array= scaler.fit_transform(data_training)


model = load_model('keras_model.h5')

data_training.tail(100)
past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index = True)
input_data = scaler.fit_transform(final_df)

x_test= []
y_test= []

for i in range(100, input_data.shape[0]):
  x_test.append(input_data[i-100:i])
  y_test.append(input_data[i,0])
x_test, y_test = np.array(x_test), np.array(y_test)

y_predicted = model.predict(x_test)

scaler = scaler.scale_
scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

st.subheader("Original vs Predicted")
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'Orginal Price')
plt.plot(y_predicted , 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
