import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# Impor Pustaka Sains Data & Deep Learning Asli
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from mlxtend.frequent_patterns import apriori, association_rules
import tensorflow as tf
import random
import warnings

warnings.filterwarnings('ignore')

# Menetapkan Seed agar hasil latihan selalu konsisten
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# ==========================================
# KONFIGURASI HALAMAN UTAMA
# ==========================================
st.set_page_config(
    page_title="Maygrillfriend DSS",
    page_icon="🥩",
    layout="wide"
)

st.title("🥩 Dashboard Pendukung Keputusan (Full Real-Time AI)")
st.subheader("Maygrillfriend Korean BBQ - Cabang Salatiga")
st.markdown("---")

# ==========================================
# SIDEBAR UNTUK UPLOAD
# ==========================================
with st.sidebar:
    st.header("📂 Manajemen Data")
    st.info("Unggah file transaksi POS Qasir (.xlsx) di bawah ini.")
    uploaded_file = st.file_uploader("Upload File Qasir", type=['xlsx'])

# ==========================================
# LOGIKA LAYAR KOSONG (SEBELUM UPLOAD)
# ==========================================
if uploaded_file is None:
    st.info("👋 Selamat datang di Sistem Pendukung Keputusan Maygrillfriend.")
    st.warning("Menunggu data... Silakan unggah file riwayat transaksi Qasir di panel sebelah kiri untuk memulai analisis AI harian.")
    st.stop() 

# ==========================================
# FUNGSI PROSES AI TOTAL (TRAINING LANGSUNG DI SERVER)
# ==========================================
@st.cache_data(show_spinner=False) 
def proses_total_ai(file_bytes):
    # 1. MEMBACA FILE EXCEL
    semua_sheet = pd.read_excel(file_bytes, engine='calamine', sheet_name=None)
    df = pd.concat(semua_sheet.values(), ignore_index=True)
    if 'Status' in df.columns:
        df = df[df['Status'] == 'Transaksi']
    df['Tanggal'] = pd.to_datetime(df['Tanggal'], format='%d-%m-%Y')

    total_baris = len(df)
    tgl_awal = df['Tanggal'].min()
    tgl_akhir = df['Tanggal'].max()

    # 2. AGREGASI DATA UNTUK LSTM
    daily_visits = df.groupby('Tanggal')['No. Struk'].nunique().reset_index()
    daily_visits.rename(columns={'No. Struk': 'Jumlah_Kunjungan'}, inplace=True)
    daily_visits = daily_visits.set_index('Tanggal').resample('D').sum()

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(daily_visits[['Jumlah_Kunjungan']])

    test_days = 30
    look_back = 14
    train_data = scaled_data[:-test_days]
    test_data = scaled_data[-(test_days + look_back):]

    def create_dataset(dataset, look_back=1):
        X, Y = [], []
        for i in range(len(dataset) - look_back):
            X.append(dataset[i:(i + look_back), 0])
            Y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(Y)

    X_train, y_train = create_dataset(train_data, look_back)
    X_test, y_test = create_dataset(test_data, look_back)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # 3. PROSES TRAINING LSTM (100 EPOCHS) RIIL
    model = Sequential()
    model.add(LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=32, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Melatih AI secara langsung di background server Streamlit
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

    # 4. PREDIKSI KELUARAN MODEL
    predicted_scaled = model.predict(X_test, verbose=0)
    pred_means = scaler.inverse_transform(predicted_scaled)
    pred_means = [max(0, round(val[0])) for val in pred_means]
    
    actual_test = daily_visits.iloc[-test_days:]['Jumlah_Kunjungan'].values
    mae_lstm = round(mean_absolute_error(actual_test, pred_means), 2)

    # Pemetaan Hasil Tanggal ke Kamus Aplikasi
    dates_test = daily_visits.index[-test_days:].strftime('%Y-%m-%d').tolist()
    pred_map = dict(zip(dates_test, pred_means))
    pred_terakhir = pred_means[-1]

    # 5. EKSTRAKSI APRIORI LIVE
    df_weekdays = df[df['Tanggal'].dt.dayofweek.isin([0, 1, 2, 3])]
    df_weekends = df[df['Tanggal'].dt.dayofweek.isin([4, 5, 6])]

    def jalankan_apriori(data_segmen):
        if data_segmen.empty: return pd.DataFrame()
        basket = (data_segmen.groupby(['No. Struk', 'Produk'])['Jumlah Produk']
                  .sum().unstack().reset_index().fillna(0).set_index('No. Struk'))
        basket_sets = basket.applymap(lambda x: 1 if x >= 1 else 0)
        
        frequent_itemsets = apriori(basket_sets, min_support=0.05, use_colnames=True)
        if frequent_itemsets.empty: return pd.DataFrame()
        
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
        rekomendasi = rules.sort_values('lift', ascending=False).head(5)
        
        rekomendasi['Jika Pelanggan Membeli'] = rekomendasi['antecedents'].apply(lambda x: ', '.join(list(x)))
        rekomendasi['Maka Cenderung Membeli'] = rekomendasi['consequents'].apply(lambda x: ', '.join(list(x)))
        df_rapi = rekomendasi[['Jika Pelanggan Membeli', 'Maka Cenderung Membeli', 'support', 'confidence', 'lift']]
        df_rapi.columns = ['Jika Pelanggan Membeli', 'Maka Cenderung Membeli', 'Tingkat Kemunculan (Support)', 'Kepastian (Confidence)', 'Kekuatan (Lift Ratio)']
        return df_rapi

    apriori_wd = jalankan_apriori(df_weekdays)
    apriori_we = jalankan_apriori(df_weekends)

    return total_baris, tgl_awal, tgl_akhir, daily_visits, mae_lstm, pred_map, pred_terakhir, apriori_wd, apriori_we

# ==========================================
# RUNNING ENGINE & VISUALISASI LOADING
# ==========================================
with st.spinner("🤖 BERKAS DITERIMA! Server sedang melatih arsitektur LSTM (100 Epochs) & menghitung matriks Apriori secara live harian... Harap tunggu sekitar 1-3 menit."):
    try:
        file_bytes = uploaded_file.getvalue()
        (total_baris, tgl_awal, tgl_akhir, daily_visits, mae_lstm, 
         pred_map, pred_terakhir, apriori_wd, apriori_we) = proses_total_ai(file_bytes)
        st.sidebar.success("✅ Model AI Berhasil Dilatih!")
    except Exception as e:
        st.error(f"Terjadi kegagalan komputasi: {e}")
        st.stop()

# ==========================================
# RENDER OUTPUT DASHBOARD
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Ringkasan Eksekutif", 
    "⚖️ Kalkulator Anti-Mubazir", 
    "📦 Strategi Temporal Bundling", 
    "📈 Simulasi Dampak Bisnis"
])

with tab1:
    st.header("Kondisi Operasional Restoran")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Total Baris Transaksi Berhasil", value=f"{total_baris:,} Baris")
    with col2:
        rentang_hari = (tgl_akhir - tgl_awal).days + 1
        st.metric(label="Rentang Waktu Operasional", value=f"{rentang_hari} Hari", delta=f"Hingga {tgl_akhir.strftime('%d %b %Y')}")
    with col3:
        st.metric(label="Akurasi Pelatihan (MAE)", value=f"± {mae_lstm} Orang", delta="Hasil Training Riil Cloud")
    
    st.markdown("### Grafik Tren Volume Kunjungan Harian")
    st.line_chart(daily_visits['Jumlah_Kunjungan'])

with tab2:
    st.header("Kalkulator Kebutuhan Bahan Baku (Prediksi LSTM)")
    st.markdown("Fitur ini membantu kepala dapur menentukan jumlah daging segar yang aman untuk dicairkan (*thawing*) harian.")
    
    col_in, col_out = st.columns([1, 2])
    with col_in:
        tgl_pilihan = st.date_input("Pilih Tanggal Operasional Besok:", tgl_akhir)
        
    with col_out:
        tgl_str = tgl_pilihan.strftime('%Y-%m-%d')
        angka_prediksi = pred_map.get(tgl_str, pred_terakhir)
            
        batas_bawah = max(0, round(angka_prediksi - mae_lstm))
        batas_atas = round(angka_prediksi + mae_lstm)
        
        st.info(f"**📋 Hasil Rekomendasi Sistem Untuk Tanggal: {tgl_pilihan.strftime('%d %B %Y')}**")
        st.markdown(f"Estimasi Kunjungan Utama: `{angka_prediksi} Kunjungan`")
        st.success(f"🟢 **Batas Aman Konservatif ({batas_bawah} Struk):** \n\nAcuan utama pencairan daging malam ini. Sangat rendah risiko dari kerugian daging sisa atau membusuk.")
        st.warning(f"🟡 **Batas Atas Antisipasi ({batas_atas} Struk):** \n\nSiapkan gas portable dan kapasitas arang ekstra untuk mengantisipasi batas atas kedatangan tamu.")

with tab3:
    st.header("Rekomendasi Paket Menu Berdasarkan Waktu")
    segmen = st.radio("Pilih Segmen Waktu Analisis:", ["Hari Kerja (Senin - Kamis)", "Akhir Pekan (Jumat - Minggu)"], horizontal=True)
    
    if segmen == "Hari Kerja (Senin - Kamis)":
        st.subheader("Pola Pembelian Hari Kerja (Karakteristik: Cepat & Hemat)")
        st.dataframe(apriori_wd, use_container_width=True)
    else:
        st.subheader("Pola Pembelian Akhir Pekan (Karakteristik: Keluarga/Rombongan)")
        st.dataframe(apriori_we, use_container_width=True)

with tab4:
    st.header("Simulasi Perencanaan Manajemen Volume")
    target_kunjungan = st.slider("Atur Target Jumlah Transaksi Restoran Minggu Depan:", 50, 300, 150)
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric(label="Estimasi Target Penjualan Paket Kombo", value=f"{round(target_kunjungan * 0.70)} Porsi", delta="Target Pramusaji Siang")
    with col_b:
        st.metric(label="Estimasi Target Penjualan Ekstra Daging", value=f"{round(target_kunjungan * 0.15)} Porsi", delta="Target Pramusaji Malam")
