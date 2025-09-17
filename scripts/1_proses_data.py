# --------------------------------------------------------------------------
# SKRIP 1: MEMPROSES DATA MENTAH (VERSI DENGAN FITUR TAMBAHAN)
# --------------------------------------------------------------------------

import pandas as pd
import numpy as np
import re
import os

print("--- Skrip Pemrosesan Data Dimulai (Versi 2) ---")

# --- BAGIAN 1: PENGATURAN PATH FILE ---
data_folder = '../data/'
file_path_input = os.path.join(data_folder, 'data.csv')
file_path_output = os.path.join(data_folder, 'data_fitur.csv')
print(f"Mencoba membaca data dari: {os.path.abspath(file_path_input)}")

# --- BAGIAN 2: MEMUAT DATA MENTAH ---
try:
    column_names = [
        'NO', 'NAMA', 'NIPD', 'JK', 'NISN', 'TEMPAT_LAHIR', 'TANGGAL_LAHIR',
        'USIA_JULI', 'JULI_BB', 'JULI_TB', 'JULI_LK', 'AGUSTUS_BB', 'AGUSTUS_TB', 'AGUSTUS_LK',
        'SEPTEMBER_BB', 'SEPTEMBER_TB', 'SEPTEMBER_LK', 'OKTOBER_BB', 'OKTOBER_TB', 'OKTOBER_LK',
        'NOPEMBER_BB', 'NOPEMBER_TB', 'NOPEMBER_LK', 'DESEMBER_BB', 'DESEMBER_TB', 'DESEMBER_LK'
    ]
    df = pd.read_csv(file_path_input, sep=';', skiprows=3, header=None, names=column_names, encoding='utf-8')
    print("✅ Data mentah 'data.csv' berhasil dimuat.")
except Exception as e:
    print(f"❌ Terjadi kesalahan saat memuat data: {e}")
    exit()

# --- BAGIAN 3: PEMBERSIHAN DATA ---
print("Membersihkan data numerik...")
numeric_cols = ['JULI_BB', 'JULI_TB', 'AGUSTUS_BB', 'AGUSTUS_TB']
for col in numeric_cols:
    if col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.replace(',', '.', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')

# --- BAGIAN 4: FEATURE ENGINEERING ---
print("Membuat fitur-fitur baru...")

# Fitur Lama
df['USIA_DALAM_BULAN'] = df['USIA_JULI'].apply(lambda u: (int(re.search(r'(\d+)\s*Tahun', u).group(1)) * 12 if re.search(r'(\d+)\s*Tahun', u) else 0) + (int(re.search(r'(\d+)\s*Bulan', u).group(1)) if re.search(r'(\d+)\s*Bulan', u) else 0) if isinstance(u, str) else None)
df['JK_ENCODED'] = df['JK'].apply(lambda x: 1 if str(x).strip() == 'L' else 0)
df['BMI_JULI'] = df['JULI_BB'] / ((df['JULI_TB'] / 100) ** 2)
df['PERUBAHAN_BB_AGUSTUS'] = df['AGUSTUS_BB'] - df['JULI_BB']

# === FITUR BARU ===
# Fitur 5: Rasio Tinggi Badan per Usia (cm/bulan)
df['TB_PER_USIA'] = df['JULI_TB'] / df['USIA_DALAM_BULAN']

# Fitur 6: Rasio Berat Badan per Tinggi Badan (kg/cm)
df['BB_PER_TB'] = df['JULI_BB'] / df['JULI_TB']

# Fitur 7: Kategori Status Gizi berdasarkan BMI
def get_status_gizi(bmi):
    if bmi < 15.5:
        return "Kurus"
    elif 15.5 <= bmi < 19.5:
        return "Normal"
    elif 19.5 <= bmi < 22.0:
        return "Gemuk"
    elif bmi >= 22.0:
        return "Obesitas"
    else:
        return "N/A" # Jika data BMI kosong
df['STATUS_GIZI_JULI'] = df['BMI_JULI'].apply(get_status_gizi)
# Catatan: Ambang batas ini adalah contoh sederhana. Klasifikasi gizi anak resmi menggunakan persentil Z-score WHO.

# --- BAGIAN 5: FINALISASI & PENYIMPANAN ---
print("Menyimpan data hasil olahan...")
kolom_fitur = [
    'NAMA', 'JK_ENCODED', 'USIA_DALAM_BULAN',
    'JULI_BB', 'JULI_TB', 'AGUSTUS_BB',
    'BMI_JULI', 'PERUBAHAN_BB_AGUSTUS',
    'TB_PER_USIA', 'BB_PER_TB', 'STATUS_GIZI_JULI' # Menambahkan kolom fitur baru
]
df_fitur = df[kolom_fitur].copy()

# Membulatkan nilai agar rapi
for col in ['BMI_JULI', 'PERUBAHAN_BB_AGUSTUS', 'TB_PER_USIA', 'BB_PER_TB']:
    if col in df_fitur.columns:
        df_fitur[col] = df_fitur[col].round(3)

df_fitur.to_csv(file_path_output, index=False)
print(f"✅ Berhasil! Data dengan fitur baru telah disimpan di: {os.path.abspath(file_path_output)}")
print("\nCuplikan 5 baris pertama data baru:")
print(df_fitur.head())
print("\n--- Skrip Selesai ---")