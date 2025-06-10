#buat file app.py
%%writefile app.py
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# === Load model dan data training ===
model = joblib.load('model_svm.pkl')
df_final = joblib.load('df_final.pkl')

# === Fungsi kategorisasi ===
def kategori_trombosit(x):
    if x < 100000:
        return 1  # Rendah
    elif x <= 150000:
        return 2  # Normal
    else:
        return 3  # Tinggi

def kategori_hemoglobin(x):
    if x < 12:
        return 1  # Rendah
    elif x <= 16.9:
        return 2  # Normal
    else:
        return 3  # Tinggi

def kategori_hematokrit(x):
    if x < 35:
        return 1  # Rendah
    elif x <= 49.9:
        return 2  # Normal
    else:
        return 3  # Tinggi

# Mapping gender jika belum ada kolom encode
if 'Jenis Kelamin' in df_final.columns and 'Jenis_kelamin' not in df_final.columns:
    label_map_gender = {'L': 1, 'P': 2, 'Laki-laki': 1, 'Perempuan': 2}
    df_final['Jenis_kelamin'] = df_final['Jenis Kelamin'].apply(lambda x: label_map_gender.get(str(x).lower().strip(), -1))

# Kolom yang digunakan
fit_columns = [
    'NO', 'Umur', 'Demam', 'Pendarahan', 'Pusing', 'Nyeri Otot/Sendi',
    'Trombosit', 'Hemoglobin', 'Hematokrit',
    'Trombosit_Kat', 'Hemoglobin_Kat', 'Hematokrit_Kat', 'Jenis_kelamin'
]

actual_fit_columns = [col for col in fit_columns if col in df_final.columns]

# Skaler
scaler = StandardScaler()
scaler.fit(df_final[actual_fit_columns])

# Diagnosis map
diagnosis_map = {1: "DD (Demam Dengue)", 2: "DBD (Demam Berdarah Dengue)", 3: "DSS (Sindrom Syok Dengue)"}

# === STREAMLIT UI ===
st.title("Prediksi Klasifikasi Demam Berdarah (SVM)")

with st.form("form_pasien"):
    No = st.number_input("Nomor Pasien", min_value=1, format="%d")
    Nama = st.text_input("Nama Pasien")
    Umur = st.number_input("Umur", min_value=0.0, step=1.0)
    Jenis_Kelamin = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])

    Demam = st.selectbox("Demam", ["YA", "TIDAK"])
    Pendarahan = st.selectbox("Pendarahan", ["YA", "TIDAK"])
    Pusing = st.selectbox("Pusing", ["YA", "TIDAK"])
    Nyeri = st.selectbox("Nyeri Otot/Sendi", ["YA", "TIDAK"])

    Trombosit = st.number_input("Trombosit", min_value=0.0)
    Hemoglobin = st.number_input("Hemoglobin (g/dL)", min_value=0.0)
    Hematokrit = st.number_input("Hematokrit (%)", min_value=0.0)

    submitted = st.form_submit_button("Prediksi")

if submitted:
    try:
        # Cek kemungkinan tertukar
        if Hemoglobin > 30 or Hematokrit < 20:
            st.warning("⚠️ Kemungkinan Hemoglobin dan Hematokrit tertukar.\n"
                       "→ Hemoglobin biasanya 10–18 g/dL\n"
                       "→ Hematokrit biasanya 30–55%")

        # Encode
        label_gender = {'laki-laki': 1, 'perempuan': 2}
        Jenis_Kelamin_encoded = label_gender[Jenis_Kelamin.lower()]

        Demam = 1 if Demam.lower() == 'ya' else 0
        Pendarahan = 1 if Pendarahan.lower() == 'ya' else 0
        Pusing = 1 if Pusing.lower() == 'ya' else 0
        Nyeri = 1 if Nyeri.lower() == 'ya' else 0

        # Kategori
        Trombosit_Kat = kategori_trombosit(Trombosit)
        Hemoglobin_Kat = kategori_hemoglobin(Hemoglobin)
        Hematokrit_Kat = kategori_hematokrit(Hematokrit)

        # Buat DataFrame input
        input_data = pd.DataFrame([[No, Umur, Demam, Pendarahan, Pusing, Nyeri,
                                    Trombosit, Hemoglobin, Hematokrit,
                                    Trombosit_Kat, Hemoglobin_Kat, Hematokrit_Kat,
                                    Jenis_Kelamin_encoded]],
                                  columns=fit_columns)

        input_data_scaled = input_data[actual_fit_columns]

        if input_data_scaled.isnull().values.any():
            st.error("Terdapat nilai kosong/NaN dalam input.")
        else:
            input_scaled = scaler.transform(input_data_scaled)
            prediction = model.predict(input_scaled)
            diagnosis = diagnosis_map.get(prediction[0], "Diagnosis tidak ditemukan")

            st.success(f"Nama Pasien: {Nama}")
            st.subheader(f"Hasil Prediksi: {diagnosis}")

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
