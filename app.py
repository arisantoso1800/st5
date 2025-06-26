import streamlit as st
import pandas as pd
import joblib
from prophet.plot import plot_plotly
from prophet import Prophet
import plotly.graph_objects as go

def retrain_and_save_model(data_path, model_path):
    # Load ulang data historis
    df = pd.read_csv(data_path)
    df['ds'] = pd.to_datetime(df['ds'])  # pastikan kolom tanggal dalam format datetime
    df['y'] = df['y'].astype(float)

    # Latih model
    model = Prophet()
    model.fit(df)

    # Simpan model
    joblib.dump(model, model_path)
    return model

st.title("Prediksi Kunjungan Pasien")

# Load model
model = joblib.load('model_kunjungan_prophet.pkl')

# Validasi apakah model sudah dilatih
if not hasattr(model, 'history') or model.history is None or model.history.empty:
    st.error("Model belum dilatih. Silakan latih model terlebih dahulu sebelum digunakan.")
    st.stop()

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
fig = plot_plotly(model, forecast)
st.plotly_chart(fig)

# Debug kecil (opsional)
st.write(forecast.head())
