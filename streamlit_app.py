

import streamlit as st
import pandas as pd
import math
from pathlib import Path
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='GDP dashboard',
    page_icon=':earth_americas:',  # This is an emoji shortcode.
)

DATA_FILENAME1 = 'data/my_data.csv'
data = pd.read_csv(DATA_FILENAME1)

st.subheader("50 Startup data")

data["Status"] = data["Status"].astype('category').cat.codes

x = data.iloc[:,:-1]
y = data.iloc[:,-1]

lr = LinearRegression()
lr.fit(x, y)  # Fit the model with the data
predictions = lr.predict(x)  # Predict using the model
r2 = r2_score(y, predictions)  # Calculate R^2 score

st.subheader(f"R^2 score: {r2}")
