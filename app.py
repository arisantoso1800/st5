import streamlit as st
import pandas as pd
import joblib
from prophet.plot import plot_plotly

# Judul
st.title("Prediksi Kunjungan Pasien")

# Load model
model = joblib.load('model_kunjungan_prophet.pkl')

# Input jumlah hari ke depan untuk prediksi
periode = st.slider("Prediksi untuk berapa hari ke depan?", min_value=7, max_value=90, value=30)

# Buat dataframe masa depan
future = model.make_future_dataframe(periods=periode)

# Prediksi
forecast = model.predict(future)

# Tampilkan data hasil prediksi
st.subheader("Hasil Prediksi")
st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periode))

# Plot interaktif
st.subheader("Grafik Prediksi")
# fig = plot_plotly(model, forecast)
# st.plotly_chart(fig)
st.write(forecast.head())

