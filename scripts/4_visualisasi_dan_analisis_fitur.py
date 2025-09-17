# --------------------------------------------------------------------------
# SKRIP 4: ANALISIS & PENYIMPANAN MODEL (VERSI PERBAIKAN FINAL)
# --------------------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import RandomForestRegressor
import joblib

print("--- Skrip Analisis dan Penyimpanan Dimulai (Versi 3 - Perbaikan) ---")

# --- BAGIAN 1: PENGATURAN PATH ---
data_folder = '../data/'
models_folder = '../models/'
file_path_input = os.path.join(data_folder, 'data_fitur.csv')
model_path_output = os.path.join(models_folder, 'random_forest_model.joblib')

os.makedirs(models_folder, exist_ok=True)

# --- BAGIAN 2: MEMUAT DATA ---
try:
    df = pd.read_csv(file_path_input)
    print("✅ File 'data_fitur.csv' berhasil dimuat.")
except FileNotFoundError:
    print("❌ GAGAL: File 'data_fitur.csv' tidak ditemukan.")
    exit()

# --- BAGIAN 3: PERSIAPAN DATA ---
print("Mempersiapkan data dengan One-Hot Encoding...")
df.dropna(inplace=True)
df_encoded = pd.get_dummies(df, columns=['STATUS_GIZI_JULI'], drop_first=True)

y = df_encoded['AGUSTUS_BB']
# === PERBAIKAN KUNCI: Menghapus fitur yang bocor ===
X = df_encoded.drop(columns=['AGUSTUS_BB', 'NAMA', 'PERUBAHAN_BB_AGUSTUS'])

print("Fitur yang digunakan (setelah perbaikan):")
print(X.columns.tolist())

# --- BAGIAN 4: MELATIH MODEL PADA SEMUA DATA ---
print("Melatih model pada semua data untuk analisis fitur...")
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X, y)

# --- BAGIAN 5: ANALISIS KEPENTINGAN FITUR ---
print("Menganalisis fitur paling penting dari model baru...")
# ... (Sisa kode analisis dan penyimpanan sama persis) ...
importances = model_rf.feature_importances_
feature_importance_df = pd.DataFrame({
    'Fitur': X.columns,
    'Tingkat_Penting': importances
}).sort_values(by='Tingkat_Penting', ascending=False)
print("\n--- FITUR PALING PENTING (MODEL BARU) ---")
print(feature_importance_df)
plt.figure(figsize=(12, 8))
sns.barplot(x='Tingkat_Penting', y='Fitur', data=feature_importance_df, palette='plasma')
plt.title('Fitur Paling Berpengaruh (Model Baru)', fontsize=16)
plt.xlabel('Tingkat Kepentingan')
plt.ylabel('Fitur')
plt.tight_layout()
plt.savefig('visualisasi_feature_importance_baru.png')
print("\n✅ Plot kepentingan fitur baru disimpan.")

# --- BAGIAN 6: MENYIMPAN MODEL BARU ---
print(f"\nMenyimpan model yang sudah diperbaiki ke: {os.path.abspath(model_path_output)}")
joblib.dump(model_rf, model_path_output)
print("✅ Model baru berhasil disimpan!")
print("\n--- Skrip Selesai ---")