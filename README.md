# Analisis Sentimen Twitter Berbahasa Indonesia: Perbandingan Arsitektur Deep Learning Berbasis Embedding IndoBERT

Repositori ini berisi empat notebook Google Colab untuk klasifikasi sentimen (**Positif** vs **Negatif**) pada data Twitter berbahasa Indonesia. Semua model memakai **IndoBERT** (`indobenchmark/indobert-base-p1`) sebagai ekstraktor embedding (bobot dibekukan), lalu dikombinasikan dengan empat arsitektur klasifikasi yang berbeda untuk dibandingkan performanya.

## Daftar Notebook

| No | File | Arsitektur |
|----|------|-----------|
| 1 | `CNN_berbasis_Embedding_IndoBERT.ipynb` | IndoBERT + CNN |
| 2 | `BiLSTM_berbasis_Embedding_IndoBERT.ipynb` | IndoBERT + BiLSTM |
| 3 | `CNN_BiLSTM_berbasis_Embedding_IndoBERT.ipynb` | IndoBERT + CNN → BiLSTM |
| 4 | `BiLSTM_CNN_berbasis_Embedding_IndoBERT.ipynb` | IndoBERT + BiLSTM → CNN |

## Detail Arsitektur

Seluruh model menerima `last_hidden_state` IndoBERT (dimensi 768) sebagai input, dengan `dropout = 0.3` dan lapisan akhir `Linear` untuk 2 kelas.

- **CNN**: tiga `Conv1d` paralel (kernel 3, 5, 7; masing-masing 128 filter) + ReLU + global max pooling, hasilnya digabung (concatenate).
- **BiLSTM**: satu lapisan LSTM dua arah (`hidden_dim = 128`) dengan mean pooling atas seluruh time-step.
- **CNN → BiLSTM**: keluaran tiga cabang CNN digabung, lalu diteruskan ke BiLSTM dengan mean pooling.
- **BiLSTM → CNN**: keluaran BiLSTM diteruskan ke tiga cabang CNN dengan global max pooling.

## Alur Eksperimen

Setiap notebook memiliki alur yang sama agar perbandingan adil:

1. **Memuat data** dari `data_twitter_final.xlsx` (kolom `normalisasi` sebagai teks dan `label`), membuang nilai kosong dan duplikat, serta memetakan label `Negatif → 0` dan `Positif → 1`.
2. **Pembagian data**: 70% latih dan 30% uji (`stratify`, `random_state=42`).
3. **Tokenisasi** dengan `BertTokenizer` IndoBERT (`max_len = 128`).
4. **Pencarian epoch optimal** memakai *Stratified 10-Fold Cross-Validation* pada data latih (maksimal 10 epoch, dipilih berdasarkan rata-rata validation loss terendah).
5. **Pelatihan akhir** pada seluruh data latih sesuai epoch optimal (Adam, `lr = 2e-4`, `batch_size = 8`, `CrossEntropyLoss`).
6. **Evaluasi** pada data uji.

## Metrik dan Visualisasi yang Dihasilkan

- Kurva validation loss hasil cross-validation (rata-rata ± standar deviasi)
- Kurva training loss dan training accuracy
- Classification report (precision, recall, F1-score)
- Confusion matrix
- Kurva ROC dan AUC
- Kurva Precision-Recall
- Visualisasi t-SNE dari fitur hasil ekstraksi
- Waktu pelatihan, waktu inferensi, dan jumlah parameter yang dapat dilatih

## Cara Menjalankan

1. Buka notebook di Google Colab (klik file `.ipynb` di GitHub, lalu gunakan *Open in Colab*, atau unggah manual ke [Google Colab](https://colab.research.google.com/)).
2. Aktifkan GPU: **Runtime → Change runtime type → T4 GPU**.
3. Siapkan file dataset `data_twitter_final.xlsx` di komputer Anda.
4. Jalankan seluruh sel secara berurutan (**Runtime → Run all**). Saat sel pertama pemuatan data berjalan, unggah `data_twitter_final.xlsx` melalui dialog yang muncul.

> **Catatan:** Proses 10-fold cross-validation memuat ulang IndoBERT pada setiap fold sehingga cukup memakan waktu. GPU sangat disarankan.

## Format Dataset

File `data_twitter_final.xlsx` minimal memiliki dua kolom:

| Kolom | Keterangan |
|-------|-----------|
| `normalisasi` | Teks tweet yang sudah dinormalisasi |
| `label` | Label sentimen: `Positif` atau `Negatif` |

## Kebutuhan Pustaka

Pustaka berikut sudah tersedia di Google Colab, kecuali `transformers` yang kadang perlu dipasang:

```
torch
transformers
numpy
pandas
matplotlib
seaborn
scikit-learn
openpyxl
```

Jika diperlukan, pasang dengan:

```bash
pip install transformers openpyxl
```

## Hasil

Isi tabel berikut setelah seluruh notebook dijalankan.

| Model | Accuracy | Precision | Recall | F1-score | AUC | Waktu Latih (s) | Parameter Terlatih |
|-------|----------|-----------|--------|----------|-----|-----------------|--------------------|
| CNN | 0.90 | 0.90 | 0.89 | 0.89 | 0.95 | 24.29 | 1475714 |
| BiLSTM | 0.88 | 0.88 | 0.89 | 0.88 | 0.95 | 24.60 | 920066 |
| CNN → BiLSTM | 0.88 | 0.89 | 0.86 | 0.87 | 0.95 | 41.64 | 2001794 |
| BiLSTM → CNN | 0.88 | 0.90 | 0.85 | 0.87 | 0.95 | 17.03 | 1412226 |

## Struktur Repositori

```
.
├── README.md
├── CNN_berbasis_Embedding_IndoBERT.ipynb
├── BiLSTM_berbasis_Embedding_IndoBERT.ipynb
├── CNN_BiLSTM_berbasis_Embedding_IndoBERT.ipynb
└── BiLSTM_CNN_berbasis_Embedding_IndoBERT.ipynb
```
