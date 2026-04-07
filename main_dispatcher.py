import sys
import os
import json
from transformers import pipeline

# Pastikan folder bisa di-import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import model 1: Panic Detector (Sistem Rule-based terjamin jalan langsung)
try:
    from Sentiment_Panic_Detection.panic_detector import analyze as panic_analyzer
    PANIC_LOADED = True
except ImportError as e:
    print(f"⚠️ Gagal memuat Panic Detector: {e}")
    PANIC_LOADED = False

# -------------------------------------------------------------
# FUNGSI PEMUATAN MODEL (Lazy Load)
# -------------------------------------------------------------
NER_PIPE = None
KLASIFIKASI_PIPE = None

def load_models():
    """Memuat model transformers dari folder jika sudah lewat proses training."""
    global NER_PIPE, KLASIFIKASI_PIPE
    
    # Load NER Model
    ner_path = "Named_Entity_Recognition/model_ner2"
    if os.path.exists(ner_path):
        print(f"[#] Memuat Model NER dari {ner_path}...")
        try:
            NER_PIPE = pipeline(
                "ner", 
                model=ner_path, 
                tokenizer=ner_path, 
                aggregation_strategy="simple"
            )
            print("[+] Model NER berhasil dimuat!")
        except Exception as e:
            print(f"[!] Error memuat NER: {e}")
    else:
        print(f"[i] Model NER di '{ner_path}' belum ada. Menjalankan versi simulasi (Mock).")

    # Load Klasifikasi Model
    klasif_path = "pesan_klasifikasi/hasil"
    if os.path.exists(klasif_path):
        print(f"[#] Memuat Model Klasifikasi dari {klasif_path}...")
        try:
            KLASIFIKASI_PIPE = pipeline(
                "text-classification", 
                model=klasif_path, 
                tokenizer=klasif_path
            )
            print("[+] Model Klasifikasi berhasil dimuat!")
        except Exception as e:
            print(f"[!] Error memuat Klasifikasi: {e}")
    else:
        print(f"[i] Model Klasifikasi di '{klasif_path}' belum ada. Menjalankan versi simulasi (Mock).")


# -------------------------------------------------------------
# FUNGSI UTAMA PENGGABUNGAN
# -------------------------------------------------------------
def proses_pesan(teks_pesan: str):
    print(f"\n[>] Memproses Pesan: '{teks_pesan}'")
    print("-" * 50)
    
    hasil_akhir = {
        "pesan_asli": teks_pesan,
        "kategori": "TIDAK DIKETAHUI",
        "tingkat_kepanikan": "TENANG",
        "skor_panik": 0.0,
        "entitas_ditemukan": [],
        "saran": "Tunggu Validasi"
    }

    # 1. PANIC DETECTION
    if PANIC_LOADED:
        hasil_panik = panic_analyzer(teks_pesan)
        hasil_akhir["tingkat_kepanikan"] = hasil_panik.level
        hasil_akhir["skor_panik"] = hasil_panik.score
        if hasil_panik.dispatch:
            hasil_akhir["saran"] = "KIRIM RELAWAN SEGERA (DISPATCH)"

    # 2. NER (Lokasi, Nama, Cedera)
    if NER_PIPE is not None:
        pred_ner = NER_PIPE(teks_pesan)
        # Bersihkan format dari HuggingFace
        entitas_bersih = []
        for ent in pred_ner:
            entitas_bersih.append({
                "jenis": ent.get('entity_group', 'UNKNOWN'),
                "kata": ent.get('word', ''),
                "akurasi": round(ent.get('score', 0.0), 2)
            })
        hasil_akhir["entitas_ditemukan"] = entitas_bersih
    else:
        # Mocking jika model belum jadi
        hasil_akhir["entitas_ditemukan"] = [{"jenis": "MOCK_LOC", "kata": "Simulasi Lokasi", "akurasi": 0.99}]

    # 3. KLASIFIKASI PESAN
    if KLASIFIKASI_PIPE is not None:
        pred_klasif = KLASIFIKASI_PIPE(teks_pesan)
        if pred_klasif:
            hasil_akhir["kategori"] = pred_klasif[0].get('label', 'UNKNOWN')
    else:
         hasil_akhir["kategori"] = "SIMULASI_KATEGORI (Model belum dilatih)"

    return hasil_akhir


if __name__ == "__main__":
    # Inisialisasi/Load AI Models ke Memory
    print("="*60)
    print("[INIT] SOVEREIGN-DISPATCH MAIN ENGINE")
    print("="*60)
    load_models()
    
    print("="*60)
    print("        KETIK 'exit' atau 'keluar' UNTUK BERHENTI")
    print("="*60)
    
    while True:
        try:
            print("\n" + "="*50)
            pesan_input = input("🗣️ Masukkan Pesan Darurat: ").strip()
            
            if pesan_input.lower() in ["exit", "keluar", "quit"]:
                print("👋 Program dihentikan. Sampai jumpa!")
                break
                
            if not pesan_input:
                continue

            hasil = proses_pesan(pesan_input)
            
            # Print hasil dalam format yang rapih
            print(json.dumps(hasil, indent=4, ensure_ascii=False))
            
        except KeyboardInterrupt:
            print("\n👋 Program dihentikan paksa.")
            break
        except Exception as e:
            print(f"❌ Terjadi kesalahan: {e}")
