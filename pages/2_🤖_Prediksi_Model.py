import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(page_title="Prediksi Model", page_icon="🤖")

st.markdown("# 🤖 Halaman Prediksi Model (Diperbarui)")
st.sidebar.header("Prediksi Model")

# --- Memuat Model ---
try:
    model = joblib.load('models/random_forest_model.joblib')
except FileNotFoundError:
    st.error("File model tidak ditemukan. Pastikan skrip 4 sudah dijalankan.")
    st.stop()

# --- Fungsi Bantuan ---
def get_status_gizi(bmi):
    if bmi < 15.5: return "Kurus"
    elif 15.5 <= bmi < 19.5: return "Normal"
    elif 19.5 <= bmi < 22.0: return "Gemuk"
    elif bmi >= 22.0: return "Obesitas"
    else: return "N/A"

# --- Form Input ---
with st.form("prediction_form"):
    st.subheader("Masukkan Data Anak:")
    col1, col2 = st.columns(2)
    with col1:
        usia_bulan = st.number_input("Usia (bulan)", min_value=1, max_value=120, value=60, step=1)
        bb_juli = st.number_input("Berat Badan Juli (kg)", min_value=1.0, max_value=50.0, value=18.0, step=0.1)
    with col2:
        jenis_kelamin = st.selectbox("Jenis Kelamin", ("Laki-laki", "Perempuan"))
        tb_juli = st.number_input("Tinggi Badan Juli (cm)", min_value=50.0, max_value=200.0, value=105.0, step=0.5)
    
    submit_button = st.form_submit_button(label='Prediksi Berat Badan')

# --- Logika Prediksi ---
if submit_button:
    # 1. Menghitung fitur turunan
    jk_encoded = 1 if jenis_kelamin == "Laki-laki" else 0
    bmi_juli = bb_juli / ((tb_juli / 100) ** 2)
    status_gizi_juli = get_status_gizi(bmi_juli)
    tb_per_usia = tb_juli / usia_bulan
    bb_per_tb = bb_juli / tb_juli
    
    st.subheader("Informasi Tambahan dari Input:")
    st.metric(label="BMI Bulan Juli", value=f"{bmi_juli:.2f}")
    st.info(f"Kategori Status Gizi: **{status_gizi_juli}**")

    # 2. Membuat DataFrame input
    input_data = {
        'JK_ENCODED': [jk_encoded], 'USIA_DALAM_BULAN': [usia_bulan],
        'JULI_BB': [bb_juli], 'JULI_TB': [tb_juli], 'BMI_JULI': [bmi_juli],
        'TB_PER_USIA': [tb_per_usia], 'BB_PER_TB': [bb_per_tb],
        'STATUS_GIZI_JULI': [status_gizi_juli]
    }
    input_df = pd.DataFrame(input_data)
    
    # 3. One-Hot Encoding
    input_df_encoded = pd.get_dummies(input_df, columns=['STATUS_GIZI_JULI'])
    
    # 4. Menyelaraskan Kolom dengan Model yang BENAR
    kolom_pelatihan = model.feature_names_in_
    
    for col in kolom_pelatihan:
        if col not in input_df_encoded.columns:
            input_df_encoded[col] = 0
    
    input_df_encoded = input_df_encoded[kolom_pelatihan]
    
    # 5. Prediksi
    prediksi = model.predict(input_df_encoded)
    hasil_prediksi = prediksi[0]

    # 6. Menampilkan Hasil
    st.subheader("🎉 Hasil Prediksi")
    st.success(f"**Prediksi Berat Badan Anak di Bulan Agustus adalah: {hasil_prediksi:.2f} kg**")
    st.metric(
        label="Berat Badan Prediksi (kg)",
        value=f"{hasil_prediksi:.2f}",
        delta=f"{(hasil_prediksi - bb_juli):.2f} kg dari bulan Juli"
    )