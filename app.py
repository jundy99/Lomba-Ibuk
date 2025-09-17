import streamlit as st

st.set_page_config(
    page_title="Prediksi Gizi Anak",
    page_icon="👶"
)

st.title("👶 Aplikasi Prediksi Gizi Anak")
st.write(
    "Selamat datang di aplikasi machine learning sederhana untuk memprediksi "
    "berat badan anak berdasarkan data bulan sebelumnya."
)

st.sidebar.success("Silakan pilih halaman di atas.")

st.info(
    "👈 **Gunakan menu di sidebar** untuk beralih ke halaman Analisis Data "
    "atau halaman Prediksi Model."
)

st.markdown(
    """
    **Tujuan Aplikasi:**
    - **Analisis Data**: Menampilkan visualisasi dan data yang digunakan.
    - **Prediksi Model**: Memberikan prediksi berat badan anak secara interaktif.
    
    Aplikasi ini dibangun menggunakan model **Random Forest** yang telah dilatih
    pada data historis.
    """
)