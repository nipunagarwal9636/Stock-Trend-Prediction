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
