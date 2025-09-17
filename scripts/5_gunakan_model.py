# --------------------------------------------------------------------------
# SKRIP 5: MENGGUNAKAN MODEL YANG SUDAH DISIMPAN
# Tujuan: Memuat model yang sudah dilatih dan menggunakannya untuk
#         membuat prediksi pada data baru.
# --------------------------------------------------------------------------

import joblib
import numpy as np
import os

print("--- Skrip Prediksi Menggunakan Model Tersimpan ---")

# --- BAGIAN 1: PENGATURAN PATH ---
# Kita perlu memberitahu skrip di mana file model disimpan
models_folder = '../models/'
model_path = os.path.join(models_folder, 'random_forest_model.joblib')

# --- BAGIAN 2: MEMUAT MODEL ---
try:
    print(f"Mencari model di: {os.path.abspath(model_path)}")
    model = joblib.load(model_path)
    print("✅ Model 'random_forest_model.joblib' berhasil dimuat!")
except FileNotFoundError:
    print(f"❌ GAGAL: File model tidak ditemukan!")
    print("Pastikan Anda sudah berhasil menjalankan skrip '4_visualisasi_dan_analisis_fitur.py'.")
    exit()

# --- BAGIAN 3: MEMBUAT FUNGSI PREDIKSI ---
# Kita bungkus logika prediksi dalam sebuah fungsi agar rapi
def prediksi_berat_badan(usia_bulan, jenis_kelamin, bb_juli, tb_juli):
    """
    Fungsi untuk memprediksi berat badan Agustus menggunakan model yang sudah dimuat.
    - jenis_kelamin: 'L' untuk Laki-laki, 'P' untuk Perempuan
    """
    print("\n--- Membuat Prediksi Baru ---")
    
    # Mengubah input menjadi format yang dimengerti model
    # 1. Encoding jenis kelamin ('L' -> 1, 'P' -> 0)
    jk_encoded = 1 if jenis_kelamin.upper() == 'L' else 0
    
    # 2. Menyiapkan data dalam bentuk array numpy 2D
    # Model mengharapkan input dalam format [[fitur1, fitur2, ...]]
    data_input = np.array([[usia_bulan, jk_encoded, bb_juli, tb_juli]])
    
    print(f"Data Input: Usia={usia_bulan} bln, JK={jenis_kelamin}, BB Juli={bb_juli} kg, TB Juli={tb_juli} cm")
    
    # 3. Melakukan prediksi
    hasil_prediksi = model.predict(data_input)
    
    # 4. Mengembalikan hasil prediksi
    return hasil_prediksi[0]

# --- BAGIAN 4: CONTOH PENGGUNAAN ---
# Mari kita coba prediksi data untuk dua anak baru yang fiktif

# Anak 1: Laki-laki, 5 tahun (60 bulan), BB 18 kg, TB 105 cm
prediksi_anak_1 = prediksi_berat_badan(usia_bulan=60, jenis_kelamin='L', bb_juli=18, tb_juli=105)
print(f"Hasil Prediksi Berat Badan Agustus untuk Anak 1: {prediksi_anak_1:.2f} kg")

# Anak 2: Perempuan, 4.5 tahun (54 bulan), BB 16.5 kg, TB 100 cm
prediksi_anak_2 = prediksi_berat_badan(usia_bulan=54, jenis_kelamin='P', bb_juli=16.5, tb_juli=100)
print(f"Hasil Prediksi Berat Badan Agustus untuk Anak 2: {prediksi_anak_2:.2f} kg")

print("\n--- Skrip Selesai ---")