import json
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from collections import Counter
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from datasets import Dataset
from seqeval.metrics import classification_report as seq_report
from seqeval.metrics import f1_score as seq_f1

warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'✅ Device: {device}')
if device == 'cuda':
    print(f'   GPU: {torch.cuda.get_device_name(0)}')

DATA_PATH = 'ner_dataset_v2.json'
# We have ner_dataset.json, wait the code uses ner_dataset_v2.json, but the folder only has ner_dataset.json.
# Let's fix that!
if not os.path.exists(DATA_PATH):
    DATA_PATH = 'ner_dataset.json'

with open(DATA_PATH, 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

print(f'Total kalimat: {len(raw_data)}')

# Statistik label
all_labels = [lbl for sent in raw_data for lbl in sent['labels']]
label_counts = Counter(all_labels)
print('\nDistribusi label IOB2:')
for lbl, cnt in sorted(label_counts.items()):
    print(f'  {lbl:12s}: {cnt}')

# ── Definisi semua label yang mungkin ─────────────────────────────────────
LABEL_LIST = [
    'O',
    'B-LOC',   'I-LOC',
    'B-NAME',  'I-NAME',
    'B-INJURY','I-INJURY',
]
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for i, l in enumerate(LABEL_LIST)}

# ── Encode label ke integer ────────────────────────────────────────────────
all_tokens = [sent['tokens'] for sent in raw_data]
all_ids    = [[LABEL2ID[l] for l in sent['labels']] for sent in raw_data]

def get_entity_key(label_ids):
    lbls = set(ID2LABEL[i] for i in label_ids)
    has_loc  = 'B-LOC' in lbls
    has_name = 'B-NAME' in lbls
    has_inj  = 'B-INJURY' in lbls
    return f'{int(has_loc)}{int(has_name)}{int(has_inj)}'

strat_keys = [get_entity_key(l) for l in all_ids]

indices = list(range(len(all_tokens)))
idx_train, idx_temp = train_test_split(
    indices, test_size=0.30, stratify=strat_keys, random_state=42)
strat_temp = [strat_keys[i] for i in idx_temp]
idx_val, idx_test = train_test_split(
    idx_temp, test_size=0.50, stratify=strat_temp, random_state=42)

print(f'\nSplit: Train={len(idx_train)} | Val={len(idx_val)} | Test={len(idx_test)}')

MODEL_NAME = 'cahya/bert-base-indonesian-522M'
MAX_LEN    = 128

print(f'⏳ Memuat tokenizer: {MODEL_NAME} ...')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print('✅ Tokenizer siap')

def tokenize_and_align(tokens_list, labels_list):
    enc = tokenizer(
        tokens_list,
        is_split_into_words=True,
        padding='max_length',
        truncation=True,
        max_length=MAX_LEN,
    )
    aligned_labels = []
    word_ids = enc.word_ids()
    prev_word_id = None
    for word_id in word_ids:
        if word_id is None:
            aligned_labels.append(-100)
        elif word_id != prev_word_id:
            aligned_labels.append(labels_list[word_id])
        else:
            aligned_labels.append(-100)
        prev_word_id = word_id
    enc['labels'] = aligned_labels
    return enc

def make_ner_dataset(indices):
    rows = {'input_ids': [], 'attention_mask': [], 'token_type_ids': [], 'labels': []}
    for i in indices:
        enc = tokenize_and_align(all_tokens[i], all_ids[i])
        rows['input_ids'].append(enc['input_ids'])
        rows['attention_mask'].append(enc['attention_mask'])
        rows['token_type_ids'].append(enc.get('token_type_ids', [0]*MAX_LEN))
        rows['labels'].append(enc['labels'])
    return Dataset.from_dict(rows)

print('⏳ Tokenisasi dataset ...')
train_ds = make_ner_dataset(idx_train)
val_ds   = make_ner_dataset(idx_val)
test_ds  = make_ner_dataset(idx_test)

print(f'⏳ Memuat model: {MODEL_NAME} ...')
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABEL_LIST),
    id2label=ID2LABEL,
    label2id=LABEL2ID,
    ignore_mismatched_sizes=True,
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    true_labels, true_preds = [], []
    for pred_seq, label_seq in zip(preds, labels):
        sent_true, sent_pred = [], []
        for p, l in zip(pred_seq, label_seq):
            if l == -100: continue
            sent_true.append(ID2LABEL[l])
            sent_pred.append(ID2LABEL[p])
        true_labels.append(sent_true)
        true_preds.append(sent_pred)

    return {'f1': seq_f1(true_labels, true_preds, average='micro')}

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# Kita kurangi epochs jadi 3 saja untuk kecepatan testing, secara produksi bisa ditingkatkan lagi nanti.
training_args = TrainingArguments(
    output_dir                  = './hasil_training_ner',
    num_train_epochs            = 3,  
    per_device_train_batch_size = 16,
    per_device_eval_batch_size  = 32,
    warmup_steps                = 50,
    weight_decay                = 0.01,
    learning_rate               = 3e-5,
    eval_strategy               = 'epoch',
    save_strategy               = 'epoch',
    load_best_model_at_end      = True,
    metric_for_best_model       = 'f1',
    greater_is_better           = True,
    logging_dir                 = './logs_ner',
    logging_steps               = 30,
    fp16                        = (device == 'cuda'),
    report_to                   = 'none',
)

trainer = Trainer(
    model           = model,
    args            = training_args,
    train_dataset   = train_ds,
    eval_dataset    = val_ds,
    tokenizer       = tokenizer,
    data_collator   = data_collator,
    compute_metrics = compute_metrics,
)

print('🚀 Mulai training NER ...')
trainer.train()
print('✅ Training selesai!')

SAVE_DIR = './model_ner2'
os.makedirs(SAVE_DIR, exist_ok=True)

trainer.save_model(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

with open(f'{SAVE_DIR}/label_map.json', 'w') as f:
    json.dump({'label2id': LABEL2ID, 'id2label': ID2LABEL}, f, indent=2)

print(f'✅ Model disimpan di: {SAVE_DIR}/')
