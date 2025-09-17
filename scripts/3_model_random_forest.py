# --------------------------------------------------------------------------
# SKRIP 3: MELATIH MODEL (VERSI DENGAN FITUR TAMBAHAN)
# Tujuan: Melatih model dengan kumpulan fitur yang lebih kaya untuk
#         meningkatkan akurasi prediksi.
# --------------------------------------------------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import os

print("--- Skrip Pelatihan Model Dimulai (Versi 2) ---")

# --- BAGIAN 1: PENGATURAN PATH FILE ---
data_folder = '../data/'
file_path_input = os.path.join(data_folder, 'data_fitur.csv')
print(f"Mencoba membaca data fitur dari: {os.path.abspath(file_path_input)}")

# --- BAGIAN 2: MEMUAT DATA FITUR ---
try:
    df = pd.read_csv(file_path_input)
    print("✅ File 'data_fitur.csv' (dengan fitur baru) berhasil dimuat.")
except FileNotFoundError:
    print("❌ GAGAL: File 'data_fitur.csv' tidak ditemukan!")
    exit()

# --- BAGIAN 3: PERSIAPAN DATA UNTUK MODEL ---
print("Mempersiapkan data dengan fitur-fitur baru...")
df.dropna(inplace=True) # Menghapus baris dengan data kosong

# === Perubahan Kunci: One-Hot Encoding untuk Fitur Kategorikal ===
# Mengubah kolom 'STATUS_GIZI_JULI' menjadi beberapa kolom numerik (0/1)
df_encoded = pd.get_dummies(df, columns=['STATUS_GIZI_JULI'], drop_first=True)

# 1. Tentukan Target (y) -> apa yang ingin kita prediksi (tetap sama)
y = df_encoded['AGUSTUS_BB']

# 2. Tentukan Fitur (X) -> semua variabel input, termasuk fitur baru
# Kita HAPUS kolom target ('AGUSTUS_BB') dan kolom non-prediktif ('NAMA')
X = df_encoded.drop(columns=['AGUSTUS_BB', 'NAMA'])

print("Fitur yang digunakan untuk melatih model:")
print(X.columns.tolist())

# 3. Bagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\n✅ Data telah dibagi: {len(X_train)} baris untuk latihan, {len(X_test)} baris untuk pengujian.")

# --- BAGIAN 4: MEMBANGUN DAN MELATIH MODEL ---
print("Melatih model Random Forest dengan fitur baru...")
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)
print("✅ Model berhasil dilatih ulang!")

# --- BAGIAN 5: EVALUASI KINERJA MODEL ---
print("Mengevaluasi kinerja model baru...")
predictions = model_rf.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print("\n--- HASIL EVALUASI BARU ---")
print(f"Rata-rata Kesalahan Prediksi (MAE): {mae:.2f} kg")
print("Bandingkan nilai MAE ini dengan hasil sebelumnya untuk melihat apakah model lebih baik.")

print("\n--- Skrip Selesai ---")