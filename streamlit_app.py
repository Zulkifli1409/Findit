import streamlit as st
import json
import os
import sys
from transformers import pipeline

# Pastikan folder lokal bisa di-import (untuk Panic Detector)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ---- PERBAIKAN BUG HUGGING FACE CLIENT CLOSED DI WINDOWS ----
os.environ["HF_HUB_DISABLE_HTTP2"] = "1"
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "your_huggingface_token_here") # Ganti 'your_huggingface_token_here' di environment variable, bukan hardcode
# -------------------------------------------------------------

try:
    from Sentiment_Panic_Detection.panic_detector import analyze as panic_analyzer
    PANIC_LOADED = True
except ImportError:
    st.error("⚠️ Gagal memuat Panic Detector lokal.")
    PANIC_LOADED = False

# =========================================================================
# ⚙️ KONFIGURASI HUGGING FACE REPO
# =========================================================================
# Nanti ganti string di bawah dengan username/nama-model Hugging Face kamu
HF_MODEL_NER = "cahya/bert-base-indonesian-522M" 
HF_MODEL_KLASIFIKASI = "Zulkifli1409/Sovereign-Klasifikasi"
HF_TOKEN = os.environ.get("HF_TOKEN") # <-- Token autentikasi ditarik dari environment variable

# Gunakan @st.cache_resource agar model tidak di-download berkali-kali saat refresh
@st.cache_resource(show_spinner=False)
def load_hf_models():
    # Login otomatis di belakang layar untuk mencegah Limit Anonymous Request
    from huggingface_hub import login
    try:
        login(token=HF_TOKEN, add_to_git_credential=False)
    except:
        pass
        
    st.info("Mencoba mengambil model langsung dari HUGGING FACE secara penuh... 🌐")
    
    # Memaksa menarik dari Hugging Face 
    try:
        klasifikasi_pipe = pipeline("text-classification", model=HF_MODEL_KLASIFIKASI, tokenizer=HF_MODEL_KLASIFIKASI, token=HF_TOKEN)
        st.success("✅ Model Klasifikasi sukses ditarik 100% dari Hugging Face!")
    except Exception as e:
        klasifikasi_pipe = None
        st.error(f"❌ Gagal mengambil model Klasifikasi dari HF: {e}")
        
    try:
        ner_pipe = pipeline("ner", model=HF_MODEL_NER, tokenizer=HF_MODEL_NER, aggregation_strategy="simple", token=HF_TOKEN)
        st.success("✅ Model NER sukses ditarik 100% dari Hugging Face!")
    except Exception as e:
        ner_pipe = None
        st.error(f"❌ Gagal mengambil model NER dari HF: {e}")
        
    return ner_pipe, klasifikasi_pipe

# =========================================================================
# 🖼️ UI STREAMLIT
# =========================================================================
st.set_page_config(page_title="Sovereign-Dispatch", page_icon="🚑", layout="centered")

st.title("🚑 Sovereign-Dispatch AI")
st.markdown("**Sistem Cerdas Analisis Pesan Darurat (Bahasa Aceh & Indonesia)**")
st.markdown("Model AI ditarik langsung dari Hugging Face.")

with st.spinner("Mengunduh/Memuat Model dari Hugging Face... ⏳"):
    NER_PIPE, KLASIFIKASI_PIPE = load_hf_models()

# Input teks User
pesan = st.text_area("🗣️ Masukkan Pesan Darurat/Laporan:", height=100, placeholder="Contoh: Tolong pak ini ada kecelakaan beruntun di Lamteumen, anak pingsan banyak darah!")

if st.button("🚀 Analisis Pesan", type="primary"):
    if not pesan.strip():
        st.error("Harap masukkan pesan terlebih dahulu!")
    else:
        st.divider()
        
        col1, col2 = st.columns(2)
        
        # 1. PANIC DETECTION (Rule-based Lokal)
        tingkat_panik = "TENANG"
        skor_panik = 0.0
        saran = "Tunggu Validasi"
        if PANIC_LOADED:
            hasil_panik = panic_analyzer(pesan)
            tingkat_panik = hasil_panik.level
            skor_panik = hasil_panik.score
            if hasil_panik.dispatch:
                saran = "🚨 KIRIM RELAWAN SEGERA (DISPATCH) 🚨"
                
        # Menentukan warna box berdasarkan panik
        panik_color = "green"
        if tingkat_panik == "WASPADA": panik_color = "orange"
        if tingkat_panik in ["PANIK", "KRITIS"]: panik_color = "red"
        
        with col1:
            st.markdown(f"### 🌡️ Tingkat Kepanikan:\n**<span style='color:{panik_color}; font-size:24px;'>{tingkat_panik}</span>**", unsafe_allow_html=True)
            st.progress(skor_panik, text=f"Skor: {skor_panik:.2f}")

        # 2. KLASIFIKASI KATEGORI (Hugging Face)
        kategori = "DEFAULT"
        if KLASIFIKASI_PIPE:
            pred_klasif = KLASIFIKASI_PIPE(pesan)
            if pred_klasif:
                kategori = pred_klasif[0].get('label', 'UNKNOWN').upper()
        else:
            kategori = "BELUM TERSEDIA"
            
        with col2:
            st.markdown(f"### 🏷️ Kategori Kejadian:\n**<span style='color:#007BFF; font-size:24px;'>{kategori}</span>**", unsafe_allow_html=True)
            if kategori == "BELUM TERSEDIA":
                st.caption("*(Model klasifikasi belum di-set di Hugging Face)*")

        st.markdown(f"**Saran Tindakan:** {saran}")
        
        # 3. NER EXTRACTOR (Hugging Face)
        st.subheader("📍 Entitas Ditemukan (Lokasi, Nama, Cedera)")
        if NER_PIPE:
            pred_ner = NER_PIPE(pesan)
            if len(pred_ner) == 0:
                st.info("Tidak ada entitas yang terdeteksi.")
            else:
                for ent in pred_ner:
                    jenis = ent.get('entity_group', 'UNKNOWN')
                    kata = ent.get('word', '')
                    akurasi = ent.get('score', 0.0)
                    
                    # Beri warna sesuai jenis
                    if "LOC" in jenis: icon = "🌍 [LOKASI]"
                    elif "NAME" in jenis: icon = "👤 [NAMA]"
                    elif "INJURY" in jenis: icon = "🩸 [CEDERA]"
                    else: icon = f"🏷️ [{jenis}]"
                        
                    st.success(f"{icon} **{kata}** *(Akurasi: {akurasi:.0%})*")
        else:
            st.warning("Model NER dari Hugging Face belum di-set. Menampilkan data dummy.")
            st.success("🌍 [LOKASI] **Simulasi Lokasi** *(Akurasi: 99%)*")

        st.caption("Diproses oleh Sovereign-Dispatch AI")
