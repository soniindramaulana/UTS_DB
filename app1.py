
# Import library yang dibutuhkan untuk melakukan data preprocessing
import numpy as np
import pandas as pd
import pickle
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.decomposition import PCA
# from sklearn.model_selection import GridSearchCV
import streamlit as st
from streamlit_option_menu import option_menu
# from IPython.display import Image
# import IPython
# import seaborn as sns
import scipy.stats
import os
import librosa
from scipy.stats import skew, kurtosis, mode, iqr





# Membaca data dari file csv
df = pd.read_csv('hasil_statistik_pertemuan4.csv')
# Memisahkan kolom target (label) dari kolom fitur
X = df.drop(columns=['Label'])  # Kolom fitur
y = df['Label']  # Kolom target

st.write("# Ekstraksi Ciri Audio Untuk Klasifikasi Audio")

with st.sidebar:
  selected = option_menu(
      menu_title="Main Menu",
      options=["Dataset", "Prediksi"],
      default_index=0
  )


if selected == "Dataset":
    st.write('''## Dataset''')
    st.write(df)
    st.write('''Dataset ini merupakan hasil Ekstraksi Ciri Audio yang mana audio yang digunakan berasal dari website Kaggle.''')
    st.write('''Dataset ini memiliki jumlah data sebanyak 2800 dengan 22 fitur.''')
    st.write('''#### Fitur-Fitur Pada Dataset''')
    st.info('''
    Fitur yang akan digunakan adalah sebagai berikut :
    1. Mean Audio
    2. Median Audio
    3. Modus Audio
    4. Nilai maksimum Audio
    5. Nilai minimum Audio
    6. Standar deviasi Audio
    7. Nilai kemiringan (skewness) Audio
    8. Nilai Keruncingan (kurtosis) Audio
    9. Nilai kuartil bawah (Q1) Audio
    10. Nilai kuartil atas (Q3) Audio
    11. Nilai IQR Audio
    12. Mean ZCR
    13. Median ZCR
    14. Std ZCR
    15. Kurtosis ZCR
    16. Skew ZCR
    17. Mean Energy RMSE
    18. Median Energy RMSE
    19. Kurtosis Energy RMSE
    20. Std Energy RMSE
    21. Skew Energy RMSE
    22. Label
     ''')

if selected == "Prediksi":
  def calculate_statistics(audio_path):
        x, sr = librosa.load(audio_path)

        mean = np.mean(x)
        std = np.std(x)
        maxv = np.amax(x)
        minv = np.amin(x)
        median = np.median(x)
        skewness = skew(x)
        kurt = kurtosis(x)
        q1 = np.quantile(x, 0.25)
        q3 = np.quantile(x, 0.75)
        mode_v = mode(x)[0]
        iqr = q3 - q1

        zcr = librosa.feature.zero_crossing_rate(x)
        mean_zcr = np.mean(zcr)
        median_zcr = np.median(zcr)
        std_zcr = np.std(zcr)
        kurtosis_zcr = kurtosis(zcr, axis=None)
        skew_zcr = skew(zcr, axis=None)

        n = len(x)
        mean_rms = np.sqrt(np.mean(x**2) / n)
        median_rms = np.sqrt(np.median(x**2) / n)
        skew_rms = np.sqrt(skew(x**2) / n)
        kurtosis_rms = np.sqrt(kurtosis(x**2) / n)
        std_rms = np.sqrt(np.std(x**2) / n)

        return [mean, median, mode_v, maxv, minv, std, skewness, kurt, q1, q3, iqr, mean_zcr, median_zcr, std_zcr, kurtosis_zcr, skew_zcr, mean_rms, median_rms, std_rms, kurtosis_rms, skew_rms]

  uploaded_file = st.file_uploader("Pilih file audio...", type=["wav","mp3"])

  if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    st.title("Prediksi Class Data Audio Menggunakan MinMax")

    if st.button("Cek Nilai Statistik"):
        # Simpan file audio yang diunggah
        audio_path = "audio_diunggah.wav"
        with open(audio_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Hitung statistik untuk file audio yang diunggah
        statistik = calculate_statistics(audio_path)

        results = []
        result = {
            'Audio Mean': statistik[0],
            'Audio Median': statistik[1],
            'Audio Mode': statistik[2],
            'Audio Maxv': statistik[3],
            'Audio Minv': statistik[4],
            'Audio Std': statistik[5],
            'Audio Skew': statistik[6],
            'Audio Kurtosis': statistik[7],
            'Audio Q1': statistik[8],
            'Audio Q3': statistik[9],
            'Audio IQR': statistik[10],
            'ZCR Mean': statistik[11],
            'ZCR Median': statistik[12],
            'ZCR Std': statistik[13],
            'ZCR Kurtosis': statistik[14],
            'ZCR Skew': statistik[15],
            'RMS Energi Mean': statistik[16],
            'RMS Energi Median': statistik[17],
            'RMS Energi Std': statistik[18],
            'RMS Energi Kurtosis': statistik[19],
            'RMS Energi Skew': statistik[20],
        }
        results.append(result)
        df = pd.DataFrame(results)
        st.write(df)

        # Hapus file audio yang diunggah
        os.remove(audio_path)

    if st.button("Deteksi Audio"):

        # Memuat data audio yang diunggah dan menyimpannya sebagai file audio
        audio_path = "audio_diunggah.wav"
        with open(audio_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Menghitung statistik untuk file audio yang diunggah (gunakan fungsi calculate_statistics sesuai kebutuhan)
        audio_features = calculate_statistics(audio_path)
        results = []
        result = {
            'Audio Mean': audio_features[0],
            'Audio Median': audio_features[1],
            'Audio Mode': audio_features[2],
            'Audio Maxv': audio_features[3],
            'Audio Minv': audio_features[4],
            'Audio Std': audio_features[5],
            'Audio Skew': audio_features[6],
            'Audio Kurtosis': audio_features[7],
            'Audio Q1': audio_features[8],
            'Audio Q3': audio_features[9],
            'Audio IQR': audio_features[10],
            'ZCR Mean': audio_features[11],
            'ZCR Median': audio_features[12],
            'ZCR Std': audio_features[13],
            'ZCR Kurtosis': audio_features[14],
            'ZCR Skew': audio_features[15],
            'RMS Energi Mean': audio_features[16],
            'RMS Energi Median': audio_features[17],
            'RMS Energi Std': audio_features[18],
            'RMS Energi Kurtosis': audio_features[19],
            'RMS Energi Skew': audio_features[20],
        }
        results.append(result)
        data_tes = pd.DataFrame(results)

        # Memuat hasil Grid Search dan model KNN terbaik dari file pickle
        grid_search_results_filename = "knn_grid_search_results.pkl"
        with open(grid_search_results_filename, 'rb') as result_file:
            grid_search_results = pickle.load(result_file)

        # Memuat model KNN terbaik dan komponen PCA terbaik
        knn_model_terbaik = grid_search_results['knn_model_terbaik']
        pca_components_terbaik = grid_search_results['pca_components_terbaik']



        # Load the model and hyperparameters
        with open('best_knn_pca_model_zscore.pkl', 'rb') as model_file:
          saved_data2 = pickle.load(model_file)
        modelminmaxgrid = saved_data2['knn_model_terbaik']
        pcaminmaxgrid = saved_data2['pca']

        implementasi_minmaxgrid = pcaminmaxgrid.transform(data_tes.values)
        predict_label2 = modelminmaxgrid.predict(implementasi_minmaxgrid)


        predict_label2
