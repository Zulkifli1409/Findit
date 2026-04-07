# 🚑 Sovereign-Dispatch AI
**Sistem Cerdas Analisis Pesan Darurat & Ekstraksi Entitas (Bahasa Indonesia & Aceh)**

![Python](https://img.shields.io/badge/Python-3.14-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.56-FF4B4B.svg)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-AI%20Models-F7931E.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU_Optimized-EE4C2C.svg)

**Sovereign-Dispatch AI** adalah sistem kecerdasan buatan terpadu yang dirancang untuk menerima, menganalisis, dan merespons laporan darurat dari masyarakat. Sistem ini mengkombinasikan teknologi NLP Berbasis *Transformers* (*Named Entity Recognition* dan *Text Classification*) dengan *Rule-Based Panic Detector* untuk menugaskan level triase yang akurat pada setiap laporan.

Proyek ini telah dikonfigurasi agar bersahabat dengan lingkungan *Cloud* (seperti Streamlit Cloud) serta mengunduh model intinya langsung dari layanan **Hugging Face Hub** tanpa menyebabkan pembengkakan memori (*Out Of Memory*).

---

## 🏗️ Struktur Proyek

- `streamlit_app.py` : Antarmuka Web App interaktif menggunakan Streamlit.
- `main_dispatcher.py` : Skrip inti (Dispatcher) yang menggabungkan seluruh jalur kecerdasan buatan.
- `Sentiment_Panic_Detection/` : Engine *rule-based* untuk menghitung tingkat stres/kepanikan teks dari sinyal tanda baca dan *keyword*.
- `Named_Entity_Recognition/` : Modul pelacakan LOKASI, KORBAN, dan CEDERA.
- `pesan_klasifikasi/` : Folder pelatihan model klasifikasi Triase darurat (KRITIS, WASPADA, INFO).
- `upload_to_hf.py` : Alat pengunggah canggih untuk mendorong file model raksasa (>500MB) ke *Hugging Face* secara stabil di jaringan berangasan.
- `requirements.txt` : Berisi *dependencies* produksi. Dirancang khusus mengambil PyTorch CPU untuk menghindari jebolnya *RAM* Streamlit Cloud.

---

## 🧠 Dokumentasi Metodologi Model AI (Klasifikasi Triase)

Sistem ini didukung oleh kecerdasan buatan yang bertugas mengkategorikan laporan masyarakat ke dalam tiga tingkat kedaruratan. Pendekatan metodenya dirancang secara ketat sebagai berikut:

### 1. Spesifikasi Dasar Arsitektur AI
Sistem menggunakan pendekatan *Natural Language Processing* (NLP) berbasis arsitektur **Transformers**. Untuk memaksimalkan pemahaman terhadap struktur bahasa lokal (termasuk dialek Aceh dan singkatan daerah), sistem di- *fine-tuning* di atas *Pre-trained Language Model* **IndoBERT Base Phase 1 (`indobenchmark/indobert-base-p1`)**. Model bertugas melakukan klasifikasi teks (`KRITIS`, `WASPADA`, dan `INFO`) dilatih menggunakan dataset sebanyak 1.198 riwayat laporan historis.

### 2. Alur Prapemrosesan (Context-Aware Preprocessing)
Mengingat urgensi dari pesan, *preprocessing* tidak meniadakan konteks emosi:
*   **Normalisasi Singkatan:** Menggunakan *regex* untuk memetakan singkatan khas masyarakat berpanik ria (seperti *tlng*, *lsng*, *drrt*) menjadi frasa baku ("tolong", "langsung", "darurat").
*   **Retensi Sinyal Panik:** Berbeda dengan pembersihan teks standar yang mencukur seluruh tanda baca, sistem ini **mengamankan tanda seru (`!`)**, karena dalam kedaruratan operasional, tanda tersebut memberikan sinyal kepanikan yang esensial.

### 3. Strategi Pemisahan Data (Stratified Splitting)
Dataset sebesar 1.198 data dipecah menggunakan *Stratified Split* dengan rasio **70% Latih (Train), 15% Validasi, dan 15% Uji Tertutup (Test)**. Teknik *stratified* memastikan porsi sampel kaum minoritas (`KRITIS`) tetap utuh. Toleransi panjang kalimat ditetapkan di batas 128 Token (*padding/truncation*).

### 4. Penanganan Ketidakseimbangan Kelas (Focal Loss)
Data lapangan selalu timpang (Laporan `INFO` sangat merajai populasi data dibandingkan `KRITIS`). Arsitektur penalti dipertajam menggunakan **Focal Loss** (Lin et al., 2017). Dibandingkan *Cross Entropy*, model akan dipukul secara matematis dengan hukuman keras apabila gagal mendeteksi sinyal `KRITIS`.
*   **Konfigurasi:** `Gamma = 2.0`, Pembobotan (`KRITIS = 2.0`, `WASPADA = 1.5`, `INFO = 1.0`).

### 5. Parameter Pelatihan (Hyperparameters)
Proses pelatihan dikawal selama **10 Epik (Epochs)** menggunakan *optimizer AdamW*. Sistem menggunakan *Learning Rate* yang sangat konservatif (`2e-5`) beserta *Weight Decay* (`0.01`) menangkal fenomena hafalan buta (*Overfitting*).

### 6. Kriteria Keberhasilan & Metrik Evaluasi Akhir
Sistem penyelamat nyawa tidak butuh basa-basi akurasi kosong. **Target utama kelayakan rilis operasional adalah Metrik Recall untuk kelas KRITIS harus mencapai target $\ge$ 95%**. Hal ini memberikan jaminan mutlak bahwa AI ini menekan angka kealpaan laporan nyawa hingga rasio batas wajar <5%.

---

## 🛠️ Cara Instalasi (Penggunaan Lokal)

Sistem bergantung penuh pada `requirements.txt` yang sudah memangkas ukuran instalasi.

1. Buka Terminal/PowerShell. Klona aplikasi ini jika belum:
```bash
git clone https://github.com/Zulkifli1409/Findit.git
cd Findit
```

2. Instal seluruh paket library (otomatis menggunakan CPU Only Pytorch agar enteng):
```bash
pip install -r requirements.txt
```

3. Jalankan aplikasi Streamlit-nya dengan mematikan *Watcher* agar tidak memunculkan *False Warning* di layar log:
```bash
streamlit run streamlit_app.py --server.fileWatcherType none
```

---

## 🌐 Konektivitas Cloud (Streamlit Cloud & Hugging Face)
Aplikasi didesain untuk mandiri di Awan. Apabila di- *deploy* di **Streamlit Community Cloud**, ia mengunduh 500 MB otak model `Zulkifli1409/Sovereign-Klasifikasi` secara *live* melalui sistem *Caching*. Pengunduhan ini dijamin menembus *Rate Limit* karena script-nya dibekali injeksi pengecekan *Environment Variable* `HF_TOKEN`.

***Developed for Rapid Triage Emergency Systems 🚁***
