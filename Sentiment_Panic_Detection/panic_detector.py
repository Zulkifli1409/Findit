"""
panic_detector.py — Sovereign-Dispatch Model 3
Skor kepanikan 0.0–1.0 dari pesan darurat (Bahasa Indonesia + Aceh)
Arsitektur: Rule-based + Optional ML (zero-shot / fine-tuned)
"""

import re
from dataclasses import dataclass, field
from typing import Optional


# ──────────────────────────────────────────────
# KAMUS SINYAL
# ──────────────────────────────────────────────

KEYWORDS_PANIC = [
    # --- INDONESIA (Medis & Kecelakaan) ---
    'sesak', 'napas', 'jantung', 'pingsan', 'pendarahan', 'kejang', 'racun', 
    'stroke', 'ambulan', 'tabrakan', 'kecelakaan', 'terjepit', 'jurang', 
    'lindas', 'terbakar', 'ledakan', 'hanyut', 'maut', 'ngeri', 'parah', 
    'patah', 'tulang', 'retak', 'hancur', 'hangus', 'api', 'meledak', 
    'evakuasi', 'nadi', 'berhenti', 'kobra', 'bisa', 'racun', 'kejang',
    'pucat', 'biru', 'lemas', 'koma', 'segera', 'cepat', 'sekarang', 
    'lsg', 'sgr', 'darurat', 'tewas', 'korban', 'tragis', 'terperangkap',
    'tertimbun', 'runtuh', 'ambruk', 'jebol', 'gas', 'bocor', 'setrum',

    # --- INDONESIA (Bencana & Kriminal) ---
    'banjir', 'longsor', 'gempa', 'tsunami', 'badai', 'puting', 'beliung', 
    'roboh', 'begal', 'rampok', 'senjata', 'pistol', 'parang', 'bacok', 
    'tusuk', 'sandera', 'culik', 'tembak', 'serang', 'ancam', 'bunuh', 
    'massa', 'amuk', 'keroyok', 'api', 'kebakaran', 'jago', 'merah',

    # --- ACEH (Medis & Trauma) ---
    'meunapas', 'hana ek', 'sigra', 'jinoe', 'peulaku', 'karam', 'meulot', 
    'teubaka', 'hanco', 'rhot', 'meubuih', 'babah', 'teubiet', 'darah', 
    'meudarah', 'hamel', 'peuleuh', 'kap', 'mita', 'peeh', 'meureuphap',
    'mate', 'pingsan', 'reumbah', 'sepit', 'patah', 'tulang', 'retak',
    'leupah', 'keung', 'saket', 'ulee', 'bocor', 'parah', 'that',

    # --- ACEH (Bencana & Alam) ---
    'ie', 'rayeuk', 'ie naek', 'luroh', 'puteng', 'beungiong', 'kencang', 
    'krueng', 'laot', 'pante', 'apui', 'dapu', 'gas', 'strum', 'listerik', 
    'kabel', 'tiang', 'jambatan', 'putoh', 'hance', 'tanoh', 'luroh',
    'gempa', 'beungeh', 'meuluap', 'tsunami', 'putih',

    # --- ACEH (Kriminal & Konflik Satwa) ---
    'beude', 'jitham', 'jiseureurang', 'baco', 'uleu', 'piton', 'gajah', 
    'harimau', 'buya', 'rimba', 'uteuen', 'begal', 'rampok', 'jipueh',
    'jidrop', 'jitembak', 'meuklei', 'prang', 'musoh',

    # --- ACEH (Penyebutan Korban/Keluarga) ---
    'adek', 'aneuk', 'ayah', 'mak', 'ibu', 'nek', 'klg', 'ureung', 
    'lon', 'kamo', 'kamoe', 'awak', 'gampong', 'desa', 'jalan', 'rot'
]

KEYWORDS_INJURY = [
    # --- INDONESIA (Trauma Fisik) ---
    'luka', 'patah', 'pendarahan', 'pingsan', 'cedera', 'darah', 'terluka', 
    'koma', 'syok', 'tidak sadar', 'remuk', 'tergencet', 'tertimpa', 
    'terkubur', 'terseret', 'trauma', 'memar', 'robek', 'perdarahan', 
    'terbakar', 'lecet', 'kritis', 'sekarat', 'hancur', 'putus', 'lumpuh', 
    'kaku', 'dingin', 'pucat', 'kebiru-biruan', 'tertusuk', 'tertembak', 
    'terbacok', 'teriris', 'tersayat', 'terkilir', 'dislokasi', 'bengkak',
    'lebam', 'nanah', 'infeksi', 'borok', 'melepuh', 'hangus',

    # --- INDONESIA (Kondisi Organ & Sistem) ---
    'sesak', 'napas', 'asfiksia', 'kejang', 'epilepsi', 'ayan', 'busa', 
    'muntah', 'keracunan', 'overdosis', 'stroke', 'serangan jantung', 
    'henti jantung', 'nadi lemah', 'pusing hebat', 'gegar otak', 
    'pecah pembuluh', 'pendarahan dalam', 'buta', 'tuli', 'bisu', 
    'patah leher', 'patah pinggang', 'patah rusuk',

    # --- ACEH (Istilah Luka & Darah) ---
    'meudarah', 'aneuk luka', 'luka that', 'meudarah rayeuk', 'meudarah dhoe',
    'ulee bocor', 'jaroe patah', 'gaki patah', 'reunggang', 'meureuphap', 
    'leupah keung', 'mameh', 'puteuh gaki', 'putoh', 'hanco', 'reumbah',
    'rhot', 'keunong baco', 'keunong tembak', 'keunong tikam', 'meuseupet',
    'teuribeit', 'teurungku', 'hance', 'karam', 'leupah', 'luka brat',

    # --- ACEH (Kondisi Medis/Kesadaran) ---
    'hana tukeu', 'hana sadar droe', 'pingsan', 'hana meunapas', 'sesak',
    'hana sanggob', 'meubuih babah', 'meubuih', 'suum', 'seujuet', 
    'pucat', 'ka udep mate', 'nyum mate', 'meugrak', 'hana meugrak le',
    'kejang', 'gigit lidah', 'saket that', 'hana ek thon', 'teusepit',

    # --- ACEH (Spesifik Kecelakaan/Bencana) ---
    'keunong apui', 'teubaka', 'keunong strum', 'meulot', 'rhot jurang',
    'hanyut', 'jiba ie', 'keunong tanoh', 'luroh', 'tibeit', 'karam laot'
]

KEYWORDS_URGENCY = [
    # --- INDONESIA (Formal & Standar) ---
    'segera', 'cepat', 'sekarang', 'buruan', 'lekas', 'darurat', 'mendesak',
    'instan', 'kilat', 'prioritas', 'utama', 'penting', 'serius', 'bahaya',
    'genting', 'langsung', 'tanpa tunda', 'detik ini', 'hari ini', 'saat ini',

    # --- INDONESIA (Bahasa Chat / Slang / Singkatan) ---
    'sgr', 'cpt', 'skrg', 'skr', 'otw', 'cepatan', 'cepet', 'gercep', 'lsg',
    'lgsg', 'p', 'p!', 'woi', 'woy', ' urgent', 'fast', 'help', 'tolong',
    'buru', 'cepetan', 'cepetin', 'plis', 'please', 'mohon',

    # --- ACEH (Waktu & Kesegeraan) ---
    'jinoe', 'got', 'nyoe', 'jinoe that', 'sigo', 'langsong', 'beuget', 
    'bagah', 'bagah that', 'sigra', 'siat nyo', 'siat treuk', 'beubagah',
    'beubagah cit', 'hanna tunda', 'hanna le', 'tepat jinoe',

    # --- ACEH (Perintah & Seruan Darurat) ---
    'peulaku', 'peuleuh', 'tulong', 'bantu', 'peugah', 'neupeugah', 'peeh',
    'jak jinoe', 'wo jinoe', 'beutrok', 'beutrok jinoe', 'pueh', 'pueh lon',
    'beuseulamat', 'beusigeu',

    # --- ACEH (Konteks Lhokseumawe/Utara - Dialek Lokal) ---
    'beubagah hey', 'bak jinoe', 'jinoe nyo', 'laju', 'laju that', 'nyo jinoe',
    'siat nyo beutrok', 'hana jan', 'hana na le', 'darurat that',

    # --- GABUNGAN / MIXED (Sering muncul di chat WA Aceh) ---
    'tolong jinoe', 'cepat bagah', 'darurat jinoe', 'help jinoe', 'sgr bagah',
    'bantu sigo', 'lsg jinoe', 'langsung bagah', 'cpt jinoe'
]

KEYWORDS_LOCATION_HINT = [
    # --- INDONESIA (Preposisi & Spasial) ---
    'di', 'lam', 'dekat', 'sekitar', 'depan', 'belakang', 'samping', 'sebelah',
    'atas', 'bawah', 'dalam', 'luar', 'antara', 'seberang', 'ujung', 'pangkal',
    'tepi', 'pinggir', 'arah', 'menuju', 'ke', 'dari', 'lintas', 'pusat',

    # --- INDONESIA (Administratif & Infrastruktur) ---
    'gampong', 'desa', 'jalan', 'gang', 'lorong', 'jln', 'gg', 'blok', 'no',
    'km', 'kelurahan', 'kecamatan', 'kabupaten', 'kota', 'dusun', 'lingkungan',
    'kompleks', 'perumahan', 'pasar', 'simpang', 'perempatan', 'pertigaan',
    'lampu merah', 'tugu', 'jembatan', 'terminal', 'stasiun', 'bandara',
    'pelabuhan', 'halte', 'spbu', 'pom bensin', 'masjid', 'menasah', 'meunasah',
    'sekolah', 'kampus', 'pabrik', 'gudang', 'kantor', 'rs', 'puskesmas',
    'klinik', 'apotek', 'lapangan', 'taman', 'hutan', 'kebun', 'sawah',

    # --- ACEH (Geografis & Administratif Lokal) ---
    'lam', 'binieh', 'ateuh', 'yuen', 'yue', 'likot', 'muko', 'wie', 'unuen',
    'teungoh', 'luah', 'krueng', 'laot', 'pante', 'gunong', 'ranto', 'paya',
    'blang', 'mon', 'sumu', 'paret', 'lhok', 'ujong', 'kuta', 'peukan',
    'kedee', 'keudee', 'balai', 'dayah', 'pesantren', 'meunasah', 'mesjid',
    'simpangan', 'rot', 'jalan rayeuk', 'jalan ubeut', 'lhokseumawe', 'aceh',

    # --- ACEH (Spesifik Spasial Dialek) ---
    'bak', 'nibak', 'u', 'u lam', 'u darat', 'u laot', 'u gunong', 'toe',
    'toe that', 'jauah', 'rab', 'di mi', 'di yue', 'di ateueh', 'lam kuta',
    'lam gampong', 'lam uteuen', 'binieh krueng', 'binieh rot', 'simpang peuet',
    'simpang lhee', 'lhok', 'paya', 'ranto',

    # --- SINGKATAN & VARIASI CHAT ---
    'd', 'dkt', 'sktr', 'dpn', 'blkng', 'smpng', 'sblh', 'jl', 'jln', 'gg',
    'komp', 'perum', 'sp', 'simpeng', 'msjd', 'mnsah', 'pusk', 'rsud'
]


# ──────────────────────────────────────────────
# DATA CLASSES
# ──────────────────────────────────────────────

@dataclass
class PanicSignal:
    name:     str
    score:    float
    evidence: str
    weight:   float = 1.0


@dataclass
class PanicResult:
    text:         str
    score:        float          # 0.0 – 1.0
    level:        str            # TENANG / WASPADA / PANIK / KRITIS
    signals:      list           # list[PanicSignal]
    dispatch:     bool           # True = langsung kirim ke relawan
    rule_score:   float = 0.0
    ml_score:     Optional[float] = None

    def summary(self) -> str:
        lines = [
            f"[{self.level}] score={self.score:.3f}  dispatch={self.dispatch}",
            f"  Teks   : {self.text[:80]}",
        ]
        active = [s for s in self.signals if s.score > 0]
        if active:
            lines.append("  Sinyal :")
            for s in active:
                lines.append(f"    {s.name:<22} +{s.score:.2f}  ({s.evidence})")
        return "\n".join(lines)


# ──────────────────────────────────────────────
# RULE-BASED ENGINE
# ──────────────────────────────────────────────

def _rule_score(text: str) -> tuple[float, list[PanicSignal]]:
    t   = text.lower()
    sig = []

    # 1. Tanda seru
    n_excl = len(re.findall(r'!', text))
    sig.append(PanicSignal(
        "Tanda seru (!)",
        min(n_excl * 0.08, 0.24),
        f"{n_excl}x '!'"
    ))

    # 2. Rasio huruf kapital
    clean = re.sub(r'\s', '', text)
    cap_ratio = sum(1 for c in clean if c.isupper()) / max(len(clean), 1)
    sig.append(PanicSignal(
        "Huruf kapital",
        min(cap_ratio * 0.50, 0.25),
        f"{cap_ratio:.0%} kapital"
    ))

    # 3. Kata panik
    hits = [k for k in KEYWORDS_PANIC if k in t]
    sig.append(PanicSignal(
        "Kata panik",
        min(len(hits) * 0.15, 0.30),
        ", ".join(hits) if hits else "—"
    ))

    # 4. Kata cedera
    hits = [k for k in KEYWORDS_INJURY if k in t]
    sig.append(PanicSignal(
        "Kata cedera",
        min(len(hits) * 0.12, 0.24),
        ", ".join(hits) if hits else "—"
    ))

    # 5. Kata urgensi temporal
    hits = [k for k in KEYWORDS_URGENCY if k in t]
    sig.append(PanicSignal(
        "Urgensi temporal",
        min(len(hits) * 0.12, 0.18),
        ", ".join(hits) if hits else "—"
    ))

    # 6. Pengulangan karakter (tlggggg, paaanik)
    repeats = len(re.findall(r'(.)\1{2,}', t))
    sig.append(PanicSignal(
        "Char berulang",
        min(repeats * 0.08, 0.16),
        f"{repeats}x" if repeats else "—"
    ))

    # 7. Kata ALL CAPS (≥3 huruf)
    all_caps = [w for w in text.split() if len(w) >= 3 and w.isupper() and re.search(r'[A-Z]', w)]
    sig.append(PanicSignal(
        "ALL CAPS words",
        min(len(all_caps) * 0.08, 0.16),
        " ".join(all_caps[:4]) if all_caps else "—"
    ))

    # 8. Pesan sangat pendek + kata darurat (SMS panik singkat)
    words = t.split()
    if len(words) <= 5 and any(k in t for k in KEYWORDS_PANIC):
        sig.append(PanicSignal(
            "Pesan singkat darurat",
            0.10,
            f"{len(words)} kata"
        ))

    total = min(sum(s.score for s in sig), 1.0)
    return total, sig


# ──────────────────────────────────────────────
# ML ENGINE (opsional)
# ──────────────────────────────────────────────

_zshot_pipeline = None

def _load_zshot():
    global _zshot_pipeline
    if _zshot_pipeline is None:
        from transformers import pipeline
        print("[PanicDetector] Loading zero-shot model (first time)...")
        _zshot_pipeline = pipeline(
            "zero-shot-classification",
            model="joeddav/xlm-roberta-large-xnli",
        )
    return _zshot_pipeline

def _ml_score_zshot(text: str) -> float:
    pipe   = _load_zshot()
    result = pipe(text, candidate_labels=["darurat mendesak", "pesan biasa"])
    idx    = result["labels"].index("darurat mendesak")
    return float(result["scores"][idx])


_ft_pipeline = None

def load_finetuned_model(model_path: str):
    """Panggil ini setelah fine-tune selesai."""
    global _ft_pipeline
    from transformers import pipeline
    _ft_pipeline = pipeline("text-classification", model=model_path)
    print(f"[PanicDetector] Fine-tuned model loaded from {model_path}")

def _ml_score_finetuned(text: str) -> float:
    result = _ft_pipeline(text)[0]
    # Asumsi label "PANIK" atau "LABEL_1"
    if result["label"] in ("PANIK", "LABEL_1", "1"):
        return float(result["score"])
    return 1.0 - float(result["score"])


# ──────────────────────────────────────────────
# LEVEL & DISPATCH
# ──────────────────────────────────────────────

def _get_level(score: float) -> str:
    if score < 0.20: return "TENANG"
    if score < 0.40: return "WASPADA"
    if score < 0.65: return "PANIK"
    return "KRITIS"

# Threshold dispatch otomatis ke relawan
DISPATCH_THRESHOLD = 0.65


# ──────────────────────────────────────────────
# PUBLIC API
# ──────────────────────────────────────────────

def analyze(
    text:      str,
    ml_mode:   str = "none",   # "none" | "zshot" | "finetuned"
    ml_weight: float = 0.4,    # bobot ML vs rule-based (0–1)
) -> PanicResult:
    """
    Analisis kepanikan dari teks pesan darurat.

    Args:
        text:      Pesan masuk (raw string)
        ml_mode:   Mode ML yang digunakan
                   "none"      → rule-based saja
                   "zshot"     → zero-shot XLM-RoBERTa (tanpa training)
                   "finetuned" → model fine-tuned lokal
        ml_weight: Bobot ML dalam ensemble (default 0.4)

    Returns:
        PanicResult dengan score, level, signals, dispatch flag
    """
    rule, signals = _rule_score(text)
    ml = None

    if ml_mode == "zshot":
        ml = _ml_score_zshot(text)
    elif ml_mode == "finetuned":
        if _ft_pipeline is None:
            raise RuntimeError("Fine-tuned model belum dimuat. Panggil load_finetuned_model() dulu.")
        ml = _ml_score_finetuned(text)

    if ml is not None:
        final = (1 - ml_weight) * rule + ml_weight * ml
    else:
        final = rule

    final = round(min(final, 1.0), 4)

    return PanicResult(
        text       = text,
        score      = final,
        level      = _get_level(final),
        signals    = signals,
        dispatch   = final >= DISPATCH_THRESHOLD,
        rule_score = round(rule, 4),
        ml_score   = round(ml, 4) if ml is not None else None,
    )


def batch_analyze(
    messages:  list[str],
    ml_mode:   str   = "none",
    ml_weight: float = 0.4,
    top_n:     int   = 10,
) -> list[PanicResult]:
    """
    Analisis batch pesan, diurutkan dari yang paling panik.
    """
    results = [analyze(m, ml_mode, ml_weight) for m in messages]
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_n]


# ──────────────────────────────────────────────
# FINE-TUNE HELPER
# ──────────────────────────────────────────────

def prepare_finetune_dataset(
    labeled_csv: str,
    output_dir:  str = "./panic_dataset",
    test_size:   float = 0.2,
):
    """
    Siapkan dataset untuk fine-tuning dari CSV berlabel.

    Format CSV:
        text,label
        "DARURAT!! tlg ibu luka parah",1
        "halo ada yang bisa bantu saya",0

    label: 1 = panik, 0 = tidak panik
    """
    import pandas as pd
    from datasets import Dataset, DatasetDict
    from sklearn.model_selection import train_test_split
    import os

    df = pd.read_csv(labeled_csv)
    assert "text" in df.columns and "label" in df.columns

    train_df, test_df = train_test_split(df, test_size=test_size, stratify=df["label"], random_state=42)

    ds = DatasetDict({
        "train": Dataset.from_pandas(train_df.reset_index(drop=True)),
        "test":  Dataset.from_pandas(test_df.reset_index(drop=True)),
    })
    os.makedirs(output_dir, exist_ok=True)
    ds.save_to_disk(output_dir)
    print(f"Dataset disimpan ke {output_dir}")
    print(f"  Train: {len(train_df)}  Test: {len(test_df)}")
    return ds


def finetune(
    dataset_dir:  str,
    output_dir:   str   = "./panic_model",
    model_name:   str   = "indobenchmark/indobert-base-p1",
    epochs:       int   = 5,
    batch_size:   int   = 16,
    learning_rate: float = 2e-5,
):
    """
    Fine-tune IndoBERT untuk klasifikasi panik/tidak panik.
    """
    from datasets import load_from_disk
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        TrainingArguments, Trainer
    )
    import numpy as np
    from sklearn.metrics import f1_score, accuracy_score

    ds        = load_from_disk(dataset_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

    ds = ds.map(tokenize, batched=True)
    ds = ds.rename_column("label", "labels")
    ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1":       f1_score(labels, preds, average="weighted"),
        }

    args = TrainingArguments(
        output_dir          = output_dir,
        num_train_epochs    = epochs,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size  = batch_size,
        learning_rate       = learning_rate,
        evaluation_strategy = "epoch",
        save_strategy       = "epoch",
        load_best_model_at_end = True,
        metric_for_best_model  = "f1",
        logging_steps       = 20,
        report_to           = "none",
    )

    trainer = Trainer(
        model           = model,
        args            = args,
        train_dataset   = ds["train"],
        eval_dataset    = ds["test"],
        tokenizer       = tokenizer,
        compute_metrics = compute_metrics,
    )

    print(f"Fine-tuning {model_name} selama {epochs} epoch...")
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model disimpan ke {output_dir}")
    return trainer


# ──────────────────────────────────────────────
# CONTOH PENGGUNAAN
# ──────────────────────────────────────────────

if __name__ == "__main__":
    test_messages = [
        "ada orang jatuh, tolong bantu ya",
        "Tlg! ibu luka di Gampong Tibang segera!",
        "TOLONG!! bapak sy patah tulang di Jalan Diponegoro sekarang!!",
        "DARURAT!! tlg tolong ibu sy di Gampong Lambhuk luka parah SEGERA!!!",
        "aneuk hana sadar lam Gampong Pande, tulong!!!",
    ]

    print("=" * 60)
    print("PANIC DETECTOR — Rule-based mode")
    print("=" * 60)
    for msg in test_messages:
        r = analyze(msg)
        print(r.summary())
        print()

    # Contoh batch
    print("=" * 60)
    print("TOP 3 PALING PANIK")
    print("=" * 60)
    top = batch_analyze(test_messages, top_n=3)
    for i, r in enumerate(top, 1):
        print(f"{i}. [{r.level}] {r.score:.3f} — {r.text[:60]}")
