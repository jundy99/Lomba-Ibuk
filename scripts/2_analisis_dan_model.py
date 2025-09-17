import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import numpy as np

# --- 1. Memuat Data yang Sudah Diolah ---
try:
    # Memuat file CSV yang sudah berisi fitur-fitur hasil langkah sebelumnya
    df = pd.read_csv('data_fitur.csv')
    print("✅ File 'data_fitur.csv' berhasil dimuat.")
except FileNotFoundError:
    print("❌ Gagal: File 'data_fitur.csv' tidak ditemukan. Pastikan Anda sudah menjalankan skrip pertama.")
    exit()

# --- 2. Eksplorasi Data Sederhana (EDA) ---
print("\n--- Analisis Statistik Deskriptif ---")
# Menampilkan statistik dasar seperti rata-rata, standar deviasi, min, max, dll.
# Kita hanya pilih kolom numerik yang relevan untuk analisis.
print(df[['USIA_DALAM_BULAN', 'JULI_BB', 'AGUSTUS_BB', 'BMI_JULI']].describe())


# --- 3. Persiapan untuk Pemodelan ---
print("\n--- Mempersiapkan Data untuk Model ---")

# Menghapus baris yang memiliki data kosong (NaN) pada kolom yang akan digunakan
df_model = df[['USIA_DALAM_BULAN', 'JK_ENCODED', 'JULI_BB', 'JULI_TB', 'AGUSTUS_BB']].dropna()

# Memisahkan antara Fitur (X) dan Target (y)
# Fitur (X): Data yang akan kita gunakan untuk memprediksi
features = ['USIA_DALAM_BULAN', 'JK_ENCODED', 'JULI_BB', 'JULI_TB']
X = df_model[features]

# Target (y): Apa yang ingin kita prediksi
y = df_model['AGUSTUS_BB']

# Membagi data menjadi data latih (80%) dan data uji (20%)
# Data latih digunakan untuk "mengajari" model
# Data uji digunakan untuk menguji seberapa baik performa model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Data dibagi menjadi {len(X_train)} baris untuk latihan dan {len(X_test)} baris untuk pengujian.")


# --- 4. Membangun dan Melatih Model ---
print("\n--- Melatih Model Machine Learning ---")

# Kita menggunakan model Regresi Linear, salah satu model paling sederhana
model = LinearRegression()

# Melatih model menggunakan data latih
model.fit(X_train, y_train)
print("✅ Model Regresi Linear berhasil dilatih!")


# --- 5. Mengevaluasi Model ---
print("\n--- Mengevaluasi Kinerja Model ---")

# Membuat prediksi menggunakan data uji (data yang belum pernah dilihat model)
predictions = model.predict(X_test)

# Menghitung seberapa besar rata-rata kesalahan prediksi (Mean Absolute Error)
mae = mean_absolute_error(y_test, predictions)
print(f"Prediksi model memiliki rata-rata kesalahan (MAE): {mae:.2f} kg")
print("Artinya, prediksi berat badan Agustus rata-rata meleset sekitar nilai tersebut dari berat sebenarnya.")


# --- 6. Contoh Prediksi ---
print("\n--- Contoh Penggunaan Model ---")
# Membuat contoh data anak baru untuk diprediksi
# Misal: Anak usia 60 bulan, Laki-laki (1), BB Juli 20 kg, TB Juli 110 cm
contoh_anak = np.array([[60, 1, 20, 110]])

# Melakukan prediksi
prediksi_baru = model.predict(contoh_anak)
print(f"Prediksi berat badan anak tersebut di bulan Agustus adalah: {prediksi_baru[0]:.2f} kg")