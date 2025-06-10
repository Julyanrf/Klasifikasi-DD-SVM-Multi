import pandas as pd # Import pandas
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

# Ensure 'Jenis_kelamin' column exists in df_final after loading
# This assumes 'Jenis Kelamin' (with space) exists and is encoded as L/P
# If not, you need to recreate or add the 'Jenis_kelamin' column here based on available data
if 'Jenis Kelamin' in df_final.columns and 'Jenis_kelamin' not in df_final.columns:
     label_map_gender = {'L': 1, 'P': 2, 'Laki-laki': 1, 'Perempuan': 2} # Use a specific map for gender if needed
     # Apply str.lower() and str.strip() for robustness if needed
     df_final['Jenis_kelamin'] = df_final['Jenis Kelamin'].apply(lambda x: label_map_gender.get(str(x).lower().strip(), -1)) # Map existing 'Jenis Kelamin' to 'Jenis_kelamin'

# === Kolom yang digunakan untuk prediksi ===
# Make sure 'Jenis_kelamin' is included in this list if you intend to use it
fit_columns = [
    'NO', 'Umur', 'Demam', 'Pendarahan', 'Pusing', 'Nyeri Otot/Sendi',
    'Trombosit', 'Hemoglobin', 'Hematokrit',
    'Trombosit_Kat', 'Hemoglobin_Kat', 'Hematokrit_Kat', 'Jenis_kelamin'
]

# Remove any columns from fit_columns that are NOT in df_final after loading
# This is a safer approach if df_final structure might vary slightly
actual_fit_columns = [col for col in fit_columns if col in df_final.columns]
print(f"Using these columns for scaling: {actual_fit_columns}")


# === Diagnosis map berdasarkan hasil prediksi ===
diagnosis_map = {1: "DD (Demam Dengue)", 2: "DBD (Demam Berdarah Dengue)", 3: "DSS (Sindrom Syok Dengue)"}

# === Fit StandardScaler berdasarkan data training ===
scaler = StandardScaler()
# Use the actual_fit_columns which are verified to exist in df_final
scaler.fit(df_final[actual_fit_columns])

# === Fungsi prediksi mandiri ===
def self_prediction():
    print("=== INPUT DATA PASIEN ===")
    try:
        No = int(input("No: "))
        Nama = input("Nama: ")  # Tidak digunakan untuk prediksi
        Umur = float(input("Umur: "))
        Jenis_Kelamin = input("Jenis Kelamin (Laki-laki/Perempuan): ").strip()

        Demam = 1 if input("Demam (YA/TIDAK): ").strip().lower() in ['ya', '1'] else 0
        Pendarahan = 1 if input("Pendarahan (YA/TIDAK): ").strip().lower() in ['ya', '1'] else 0
        Pusing = 1 if input("Pusing (YA/TIDAK): ").strip().lower() in ['ya', '1'] else 0
        Nyeri = 1 if input("Nyeri Otot/Sendi (YA/TIDAK): ").strip().lower() in ['ya', '1'] else 0

        Trombosit = float(input("Trombosit: "))
        Hemoglobin = float(input("Hemoglobin: ").replace(",", "."))
        Hematokrit = float(input("Hematokrit: ").replace(",", "."))

        # Validasi dasar agar tidak tertukar
        if Hemoglobin > 30 or Hematokrit < 20:
            print("\n[Peringatan] Cek kembali input! Kemungkinan Hemoglobin dan Hematokrit tertukar.")
            print("→ Hemoglobin biasanya antara 10–18 g/dL")
            print("→ Hematokrit biasanya antara 30–55%")
            konfirmasi = input("Apakah yakin data sudah benar? (Y/N): ").strip().lower()
            if konfirmasi != 'y':
                print("Silakan input ulang.")
                return


        # === Kategorisasi nilai numerik ===
        Trombosit_Kat = kategori_trombosit(Trombosit)
        Hemoglobin_Kat = kategori_hemoglobin(Hemoglobin)
        Hematokrit_Kat = kategori_hematokrit(Hematokrit)

        # === Encode jenis kelamin ===
        label_jenis_kelamin_input = {'laki-laki': 1, 'perempuan': 2} # Use lowercase for input matching
        Jenis_Kelamin_encoded = label_jenis_kelamin_input.get(Jenis_Kelamin.lower(), -1) # Ensure lowercase match

        # === Bangun DataFrame input ===
        # Make sure input_data columns match the columns used for scaler.fit
        input_data = pd.DataFrame([[
            No, Umur, Demam, Pendarahan, Pusing, Nyeri,
            Trombosit, Hemoglobin, Hematokrit,
            Trombosit_Kat, Hemoglobin_Kat, Hematokrit_Kat,
            Jenis_Kelamin_encoded # Include encoded gender
        ]], columns=fit_columns) # Use the full fit_columns list here initially

        # Select only the columns used for scaling from the input data
        input_data_scaled = input_data[actual_fit_columns]


        print("\n[DEBUG] Data sebelum scaling:")
        print(input_data_scaled)

        # === Cek NaN sebelum scaling ===
        if input_data_scaled.isnull().values.any():
            print("\n[ERROR] Terdapat nilai NaN pada input:")
            print(input_data_scaled.isnull())
            return

        # === Scaling dan prediksi ===
        # Use the subset of input data for scaling
        input_scaled = scaler.transform(input_data_scaled)
        prediction = model.predict(input_scaled)

        # Mapkan hasil prediksi ke diagnosis
        diagnosis = diagnosis_map.get(prediction[0], "Diagnosis tidak ditemukan")

        print("\n=== HASIL PREDIKSI ===")
        print(f"Nama: {Nama}")
        print(f"Prediksi: {diagnosis}")

    except Exception as e:
        print(f"\n[ERROR] Terjadi kesalahan input atau proses: {e}")

# === Jalankan fungsi ===
self_prediction()
