import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import re

# Tentukan path folder model yang sudah di-training
MODEL_PATH = './hasil'

# ── Kamus normalisasi singkatan panik (Dari preprocessing Anda) ────────────
KAMUS_SINGKATAN = {
    r'\btlg\b': 'tolong',
    r'\bbantu\b': 'bantu',
    r'\blsg\b': 'langsung',
    r'\bsgr\b': 'segera',
    r'\bdrrt\b': 'darurat',
    r'\bdrt\b': 'darurat',
    r'\bpls\b': 'tolong',
    r'\bkrg\b': 'kurang',
    r'\bskt\b': 'sakit',
    r'\bukrg\b': 'aneuk',       # Aceh: anak
    r'\blkgn\b': 'lingkungan',
}

def preprocess(text: str) -> str:
    """Fungsi preprocessing yang sama dengan di fase training"""
    text = str(text)
    for pola, ganti in KAMUS_SINGKATAN.items():
        text = re.sub(pola, ganti, text, flags=re.IGNORECASE)
    text = re.sub(r'[^\w\s!?,.]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def test_model():
    print("⏳ Memuat model dan tokenizer...")
    
    # ── Menganalisis dan Mencoba Menggunakan CUDA ──────────────────────────
    # Cek ketersediaan CUDA di sistem
    if torch.cuda.is_available():
        device_id = 0
        device_name = torch.cuda.get_device_name(0)
        print(f"✅ GPU terdeteksi! Menggunakan CUDA: {device_name}")
    else:
        device_id = -1
        print("⚠️ CUDA tidak spesifik/tersedia. Menggunakan CPU.")
    
    try:
        # Load pipeline klasifikasi
        classifier = pipeline(
            'text-classification',
            model=MODEL_PATH,
            tokenizer=MODEL_PATH,
            device=device_id
        )
        print("✅ Model berhasil dimuat!\n")
    except Exception as e:
        print(f"❌ Gagal memuat model. Pastikan folder '{MODEL_PATH}' tersedia.")
        print("Error:", e)
        return

    print('\n🧪 UJI COBA INFERENSI MODEL (INTERAKTIF)')
    print("Ketik 'exit' atau 'keluar' untuk menghentikan program.")
    print('-' * 70)
    
    while True:
        try:
            pesan = input("\n📝 Masukkan pesan: ")
            if pesan.lower() in ['exit', 'keluar', 'quit']:
                print("Selesai. Terima kasih!")
                break
            
            if not pesan.strip():
                continue

            pesan_bersih = preprocess(pesan)
            hasil = classifier(pesan_bersih)[0]
            
            label = hasil['label']
            skor  = hasil['score']
            
            emoji = {'KRITIS': '🔴', 'WASPADA': '🟡', 'INFO': '🟢'}.get(label, '⚪')
            
            print(f'{emoji} [{label}] (Yakin: {skor:.2%})')
            print(f'   Preproses  : {pesan_bersih}')
            print('-' * 70)
        except KeyboardInterrupt:
            print("\nProgram dihentikan.")
            break

if __name__ == "__main__":
    test_model()
