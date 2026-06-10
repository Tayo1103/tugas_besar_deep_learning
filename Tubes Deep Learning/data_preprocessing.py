import os
import csv
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split

# ==================================================
# CONFIG
# ==================================================

DATASET_DIR  = "dataset"
IMG_DIR      = os.path.join(DATASET_DIR, "images")
CSV_FILE     = os.path.join(DATASET_DIR, "labels.csv")

OUTPUT_DIR   = "dataset_split"
TRAIN_DIR    = os.path.join(OUTPUT_DIR, "train")
VAL_DIR      = os.path.join(OUTPUT_DIR, "val")
TEST_DIR     = os.path.join(OUTPUT_DIR, "test")

IMG_SIZE     = 224  # EfficientNet-B0 default

LABEL_COLS   = ["front_left", "front_right", "rear_left", "rear_right", "hood"]

TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.20
TEST_RATIO   = 0.10

# ==================================================
# 1. LOAD & VALIDASI CSV
# ==================================================

print("=" * 50)
print("1. LOAD & VALIDASI CSV")
print("=" * 50)

df = pd.read_csv(CSV_FILE)

print(f"Total rows     : {len(df)}")
print(f"Columns        : {list(df.columns)}")
print(f"Missing values :\n{df.isnull().sum()}")

# Hapus duplikat
before = len(df)
df = df.drop_duplicates(subset="image")
print(f"\nDuplikat dihapus : {before - len(df)} rows")

# ==================================================
# 2. VALIDASI FILE GAMBAR
# ==================================================

print("\n" + "=" * 50)
print("2. VALIDASI FILE GAMBAR")
print("=" * 50)

missing_files = []
corrupt_files = []

for idx, row in df.iterrows():
    path = os.path.join(IMG_DIR, row["image"])

    # Cek file ada
    if not os.path.exists(path):
        missing_files.append(row["image"])
        continue

    # Cek file bisa dibuka
    try:
        img = Image.open(path)
        img.verify()
    except Exception:
        corrupt_files.append(row["image"])

print(f"Missing files  : {len(missing_files)}")
print(f"Corrupt files  : {len(corrupt_files)}")

# Hapus baris yang filenya tidak valid
invalid = set(missing_files + corrupt_files)
df = df[~df["image"].isin(invalid)].reset_index(drop=True)
print(f"Sisa data valid: {len(df)} rows")

# ==================================================
# 3. DISTRIBUSI LABEL
# ==================================================

print("\n" + "=" * 50)
print("3. DISTRIBUSI LABEL")
print("=" * 50)

label_counts = df[LABEL_COLS].sum()
print(label_counts)

# Plot distribusi
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart per label
label_counts.plot(kind="bar", ax=axes[0], color="steelblue", edgecolor="black")
axes[0].set_title("Jumlah Positif per Label")
axes[0].set_xlabel("Label")
axes[0].set_ylabel("Jumlah")
axes[0].set_xticklabels(LABEL_COLS, rotation=30)

for i, v in enumerate(label_counts):
    axes[0].text(i, v + 10, str(int(v)), ha="center", fontsize=9)

# Distribusi kombinasi label (berapa banyak label aktif per gambar)
df["label_count"] = df[LABEL_COLS].sum(axis=1)
df["label_count"].value_counts().sort_index().plot(
    kind="bar", ax=axes[1], color="coral", edgecolor="black"
)
axes[1].set_title("Distribusi Jumlah Label Aktif per Gambar")
axes[1].set_xlabel("Jumlah Label Aktif")
axes[1].set_ylabel("Jumlah Gambar")

plt.tight_layout()
plt.savefig(os.path.join(DATASET_DIR, "label_distribution.png"), dpi=150)
plt.show()
print("Plot disimpan → dataset/label_distribution.png")

# ==================================================
# 4. RESIZE & NORMALISASI (VERIFIKASI SAMPEL)
# ==================================================

print("\n" + "=" * 50)
print("4. CEK UKURAN & MODE GAMBAR")
print("=" * 50)

sizes   = set()
modes   = set()
samples = min(200, len(df))

for fname in df["image"].head(samples):
    path = os.path.join(IMG_DIR, fname)
    img  = Image.open(path)
    sizes.add(img.size)
    modes.add(img.mode)

print(f"Mode gambar    : {modes}")
print(f"Ukuran unik    : {sizes}")

if len(sizes) > 1:
    print("⚠ Ada gambar dengan ukuran berbeda — akan di-resize saat training")
else:
    print("✓ Semua gambar ukurannya sama")

# ==================================================
# 5. SPLIT DATASET
# ==================================================

print("\n" + "=" * 50)
print("5. SPLIT DATASET")
print("=" * 50)

# Stratify berdasarkan kombinasi label (sebagai string)
df["label_combo"] = df[LABEL_COLS].astype(str).agg("_".join, axis=1)

# Train vs (val + test)
train_df, temp_df = train_test_split(
    df,
    test_size=(VAL_RATIO + TEST_RATIO),
    random_state=42,
    stratify=df["label_combo"]
)

# Val vs Test
val_df, test_df = train_test_split(
    temp_df,
    test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
    random_state=42,
    stratify=temp_df["label_combo"]
)

train_df = train_df.reset_index(drop=True)
val_df   = val_df.reset_index(drop=True)
test_df  = test_df.reset_index(drop=True)

print(f"Train : {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
print(f"Val   : {len(val_df)}   ({len(val_df)/len(df)*100:.1f}%)")
print(f"Test  : {len(test_df)}  ({len(test_df)/len(df)*100:.1f}%)")

# ==================================================
# 6. COPY FILE KE FOLDER SPLIT
# ==================================================

print("\n" + "=" * 50)
print("6. COPY FILE KE FOLDER SPLIT")
print("=" * 50)

def copy_split(split_df, split_name, split_dir):
    img_out = os.path.join(split_dir, "images")
    os.makedirs(img_out, exist_ok=True)

    for fname in split_df["image"]:
        src = os.path.join(IMG_DIR, fname)
        dst = os.path.join(img_out, fname)
        shutil.copy2(src, dst)

    # Simpan CSV split
    csv_out = os.path.join(split_dir, "labels.csv")
    cols_to_save = ["image"] + LABEL_COLS
    split_df[cols_to_save].to_csv(csv_out, index=False)

    print(f"✓ {split_name}: {len(split_df)} gambar → {split_dir}")

copy_split(train_df, "Train", TRAIN_DIR)
copy_split(val_df,   "Val",   VAL_DIR)
copy_split(test_df,  "Test",  TEST_DIR)

# ==================================================
# 7. SUMMARY AKHIR
# ==================================================

print("\n" + "=" * 50)
print("7. SUMMARY")
print("=" * 50)

for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
    print(f"\n[{split_name}]")
    print(split_df[LABEL_COLS].sum().to_string())

print("\n ✓ Preprocessing selesai!")
print(f"Output folder  : {OUTPUT_DIR}/")
print(f"  ├── train/images/  ({len(train_df)} files)")
print(f"  ├── train/labels.csv")
print(f"  ├── val/images/    ({len(val_df)} files)")
print(f"  ├── val/labels.csv")
print(f"  ├── test/images/   ({len(test_df)} files)")
print(f"  └── test/labels.csv")