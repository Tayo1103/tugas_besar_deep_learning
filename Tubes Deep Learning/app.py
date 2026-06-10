import os
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO

import torch
import torch.nn as nn
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import streamlit as st

import json

# ==================================================
# CONFIG
# ==================================================

LABEL_COLS = [
    "front_left",
    "front_right",
    "rear_left",
    "rear_right",
    "hood"
]

LABEL_NAMES = {
    "front_left": "Pintu Depan Kiri",
    "front_right": "Pintu Depan Kanan",
    "rear_left": "Pintu Belakang Kiri",
    "rear_right": "Pintu Belakang Kanan",
    "hood": "Kap Mesin",
}

NUM_CLASSES = len(LABEL_COLS)
IMG_SIZE = 224
CHECKPOINT = "best_model.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BEST_THRESHOLDS = np.array([0.60, 0.50, 0.75, 0.40, 0.50], dtype=np.float32)


# ==================================================
# MODEL
# ==================================================

class CarPartClassifier(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super().__init__()

        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=0
        )

        in_features = self.backbone.num_features

        self.head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat)


@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(CHECKPOINT):
        st.error(
            f"File checkpoint `{CHECKPOINT}` tidak ditemukan. "
            "Pastikan file `best_model.pth` berada dalam folder yang sama dengan `app.py`."
        )
        st.stop()

    model = CarPartClassifier(
        num_classes=NUM_CLASSES,
        pretrained=False
    )

    try:
        checkpoint = torch.load(
            CHECKPOINT,
            map_location=DEVICE,
            weights_only=True
        )
    except TypeError:
        checkpoint = torch.load(
            CHECKPOINT,
            map_location=DEVICE
        )

    # Jika checkpoint disimpan sebagai dictionary lengkap
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Antisipasi jika model pernah disimpan dengan DataParallel
    state_dict = {
        key.replace("module.", ""): value
        for key, value in state_dict.items()
    }

    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE)
    model.eval()

    return model


# ==================================================
# TRANSFORM INFERENCE
# Sama dengan val_transform / test_transform saat training
# ==================================================

transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),

    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),

    ToTensorV2()
])


def predict(model, img: Image.Image):
    img_rgb = img.convert("RGB")
    arr = np.array(img_rgb)

    tensor = transform(image=arr)["image"]
    tensor = tensor.unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        logits = model(tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    preds = (probs > BEST_THRESHOLDS).astype(int)

    return probs, preds


# ==================================================
# UI CONFIG
# ==================================================

st.set_page_config(
    page_title="Car Part State Detection",
    page_icon="🚗",
    layout="wide"
)


# ==================================================
# HEADER
# ==================================================

st.markdown("""
<style>
    .main-title {
        text-align: center;
        padding: 28px 20px 18px 20px;
        border-radius: 22px;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 45%, #00c6ff 100%);
        color: white;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.12);
        margin-bottom: 18px;
    }

    .main-title h1 {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 8px;
        letter-spacing: 0.5px;
    }

    .main-title p {
        font-size: 16px;
        color: #eef6ff;
        margin: 0;
    }

    .subtitle-badge {
        display: inline-block;
        margin-top: 12px;
        padding: 6px 14px;
        border-radius: 999px;
        background-color: rgba(255, 255, 255, 0.18);
        color: #ffffff;
        font-size: 13px;
        font-weight: 500;
    }
    
    div.stButton > button {
        background: linear-gradient(135deg, #1E88E5, #1565C0);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1rem;
        font-size: 16px;
        font-weight: 700;
        box-shadow: 0 6px 14px rgba(21, 101, 192, 0.25);
        transition: all 0.2s ease-in-out;
    }

    div.stButton > button:hover {
        background: linear-gradient(135deg, #1565C0, #0D47A1);
        color: white;
        border: none;
        transform: translateY(-1px);
        box-shadow: 0 8px 18px rgba(13, 71, 161, 0.32);
    }

    div.stButton > button:focus {
        color: white;
        border: none;
        box-shadow: 0 0 0 0.2rem rgba(30, 136, 229, 0.35);
    }
</style>

<div class="main-title">
    <h1>🚗 Car Part State Detection</h1>
    <p>Sistem Deteksi Status Komponen Mobil Berbasis Multi-Label Classification</p>
    <div class="subtitle-badge">
        EfficientNet-B0 · Deep Learning · Image Classification
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()


# ==================================================
# LOAD MODEL
# ==================================================

with st.spinner("Memuat model..."):
    model = load_model()


# ==================================================
# TABS
# ==================================================

tab1, tab2 = st.tabs([
    "🔍 Deteksi Gambar",
    "📊 Hasil Uji Robustness"
])


# ==================================================
# TAB 1: DETEKSI GAMBAR
# ==================================================

with tab1:

    st.markdown("### 🔍 Upload Gambar Mobil")

    st.info(
        "⚠️ **Pastikan gambar diambil dari simulator 3D berikut:**\n\n"
        "🔗 http://103.233.100.26:8080/\n\n"
        "Model dilatih khusus pada gambar dari simulator tersebut. "
        "Gambar dari sumber lain dapat menghasilkan prediksi yang tidak akurat.",
        icon="ℹ️"
    )

    uploaded = st.file_uploader(
        "Pilih gambar (.png / .jpg / .jpeg)",
        type=["png", "jpg", "jpeg"]
    )

    # Reset hasil prediksi jika file yang di-upload berubah
    if "uploaded_filename" not in st.session_state:
        st.session_state.uploaded_filename = None

    if "probs" not in st.session_state:
        st.session_state.probs = None

    if "preds" not in st.session_state:
        st.session_state.preds = None

    if uploaded is not None:

        if st.session_state.uploaded_filename != uploaded.name:
            st.session_state.uploaded_filename = uploaded.name
            st.session_state.probs = None
            st.session_state.preds = None

        img = Image.open(uploaded).convert("RGB")

        col1, col2 = st.columns([1, 1], gap="large")

        # =========================
        # KOLOM KIRI: GAMBAR
        # =========================
        with col1:
            st.markdown("#### 🖼️ Gambar Input")
            st.image(img, use_container_width=True)

        # =========================
        # KOLOM KANAN: TOMBOL + HASIL
        # =========================
        with col2:
            st.markdown("#### ⚙️ Kontrol Deteksi")

            st.markdown(
                """
                Gambar sudah berhasil di-upload.  
                Klik tombol di bawah untuk menjalankan proses deteksi.
                """
            )

            detect_button = st.button(
                "🚀 Mulai Deteksi",
                type="primary",
                use_container_width=True
            )

            if detect_button:
                with st.spinner("Memproses gambar..."):
                    probs, preds = predict(model, img)

                st.session_state.probs = probs
                st.session_state.preds = preds

            # ==================================================
            # HASIL PREDIKSI
            # Sekarang tampil tepat di bawah tombol deteksi
            # ==================================================

            if st.session_state.probs is None or st.session_state.preds is None:
                st.warning(
                    "Prediksi belum dijalankan. Klik tombol **Mulai Deteksi** "
                    "untuk melihat hasil klasifikasi."
                )

            else:
                probs = st.session_state.probs
                preds = st.session_state.preds

                st.divider()
                st.markdown("### ✅ Hasil Prediksi")

                for i, label_col in enumerate(LABEL_COLS):
                    part_name = LABEL_NAMES[label_col]
                    prob = float(probs[i])
                    pred = int(preds[i])
                    threshold = float(BEST_THRESHOLDS[i])

                    if pred == 1:
                        status_text = "TERBUKA"
                        status_icon = "🟢"
                        card_bg = "#eafaf1"
                        border_color = "#2ecc71"
                    else:
                        status_text = "TERTUTUP"
                        status_icon = "⚪"
                        card_bg = "#f4f6f7"
                        border_color = "#95a5a6"

                    st.markdown(
                        f"""
                        <div style="
                            padding: 14px 16px;
                            border-radius: 14px;
                            border: 1.5px solid {border_color};
                            background-color: {card_bg};
                            margin-bottom: 12px;
                        ">
                            <div style="
                                display: flex;
                                justify-content: space-between;
                                align-items: center;
                            ">
                                <div>
                                    <div style="font-size: 16px; font-weight: 700;">
                                        {part_name}
                                    </div>
                                    <div style="font-size: 13px; color: #555; margin-top: 4px;">
                                        Probabilitas: {prob:.4f} | Threshold: {threshold:.2f}
                                    </div>
                                </div>
                                <div style="
                                    font-size: 15px;
                                    font-weight: 800;
                                    text-align: right;
                                ">
                                    {status_icon}<br>{status_text}
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                st.divider()

                st.markdown("### 📌 Ringkasan Deteksi")

                open_parts = [
                    LABEL_NAMES[LABEL_COLS[i]]
                    for i in range(NUM_CLASSES)
                    if preds[i] == 1
                ]

                if open_parts:
                    st.success(
                        "**Bagian yang terdeteksi terbuka:**\n\n" +
                        "\n".join(f"- {part}" for part in open_parts)
                    )
                else:
                    st.success("✅ Semua bagian terdeteksi dalam kondisi **tertutup**.")

                with st.expander("📈 Lihat probabilitas detail"):
                    detail_df = pd.DataFrame({
                        "Label": [LABEL_NAMES[col] for col in LABEL_COLS],
                        "Probabilitas": [round(float(p), 4) for p in probs],
                        "Threshold": [round(float(t), 2) for t in BEST_THRESHOLDS],
                        "Prediksi": [
                            "Terbuka" if int(preds[i]) == 1 else "Tertutup"
                            for i in range(NUM_CLASSES)
                        ]
                    })

                    st.dataframe(
                        detail_df,
                        use_container_width=True,
                        hide_index=True
                    )

    else:
        st.warning("Silakan upload gambar terlebih dahulu sebelum menjalankan deteksi.")


# ==================================================
# TAB 2: HASIL UJI ROBUSTNESS
# ==================================================

with tab2:

    st.markdown("### 📊 Hasil Uji Robustness Model")
    st.markdown(
        "Berikut rangkuman performa model pada **test set**. "
        "Nilai metrik dapat diisi sesuai hasil evaluasi akhir dari notebook training."
    )

    st.divider()

    # ==================================================
    # METRIK UTAMA
    # ==================================================

    st.markdown("#### 🎯 Metrik Keseluruhan")

    m1, m2, m3, m4 = st.columns(4)

    if os.path.exists("overall_metrics.json"):
        with open("overall_metrics.json", "r") as f:
            overall_metrics = json.load(f)

        m1.metric("Macro F1", f"{overall_metrics['macro_f1']:.4f}")
        m2.metric("Micro F1", f"{overall_metrics['micro_f1']:.4f}")
        m3.metric("Macro Precision", f"{overall_metrics['macro_precision']:.4f}")
        m4.metric("Macro Recall", f"{overall_metrics['macro_recall']:.4f}")

    else:
        m1.metric("Macro F1", "—")
        m2.metric("Micro F1", "—")
        m3.metric("Macro Precision", "—")
        m4.metric("Macro Recall", "—")

        st.warning(
            "File `overall_metrics.json` belum ditemukan. "
            "Simpan hasil evaluasi dari notebook training terlebih dahulu."
        )

    st.divider()

    # ==================================================
    # PERFORMA PER LABEL
    # ==================================================

    st.markdown("#### 📋 Performa Per Label")

    if os.path.exists("classification_report.csv"):
        df_report = pd.read_csv("classification_report.csv", index_col=0)

        rows = []
        for label_col in LABEL_COLS:
            rows.append({
                "Label": LABEL_NAMES[label_col],
                "Precision": round(df_report.loc[label_col, "precision"], 4),
                "Recall": round(df_report.loc[label_col, "recall"], 4),
                "F1-Score": round(df_report.loc[label_col, "f1-score"], 4),
                "Support": int(df_report.loc[label_col, "support"]),
            })

        df_metrics = pd.DataFrame(rows)

    else:
        df_metrics = pd.DataFrame({
            "Label": list(LABEL_NAMES.values()),
            "Precision": ["—"] * NUM_CLASSES,
            "Recall": ["—"] * NUM_CLASSES,
            "F1-Score": ["—"] * NUM_CLASSES,
            "Support": ["—"] * NUM_CLASSES,
        })

        st.warning(
            "File `classification_report.csv` belum ditemukan. "
            "Tabel performa per label belum dapat ditampilkan."
        )

    st.dataframe(
        df_metrics,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # ==================================================
    # VISUALISASI HASIL EVALUASI
    # ==================================================

    st.markdown("#### 🖼️ Visualisasi Hasil Evaluasi")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Training Curve**")
        if os.path.exists("training_curve.png"):
            st.image("training_curve.png", use_container_width=True)
        else:
            st.warning(
                "File `training_curve.png` belum ditemukan. "
                "Simpan hasil grafik training curve dengan nama tersebut."
            )

    with col2:
        st.markdown("**Confusion Matrix per Label**")
        if os.path.exists("confusion_matrix.png"):
            st.image("confusion_matrix.png", use_container_width=True)
        else:
            st.warning(
                "File `confusion_matrix.png` belum ditemukan. "
                "Simpan hasil confusion matrix dengan nama tersebut."
            )

    with col3:
        st.markdown("**Sampel Prediksi**")
        if os.path.exists("sample_predictions.png"):
            st.image("sample_predictions.png", use_container_width=True)
        else:
            st.warning(
                "File `sample_predictions.png` belum ditemukan. "
                "Simpan visualisasi sampel prediksi dengan nama tersebut."
            )

    st.divider()

    # ==================================================
    # THRESHOLD PER LABEL
    # ==================================================

    st.markdown("#### ⚙️ Threshold Optimal per Label")

    df_thresh = pd.DataFrame({
        "Label": list(LABEL_NAMES.values()),
        "Kolom Label": LABEL_COLS,
        "Threshold": BEST_THRESHOLDS
    })

    st.dataframe(
        df_thresh,
        use_container_width=True,
        hide_index=True
    )


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.markdown(
    "<p style='text-align:center; color:gray; font-size:13px'>"
    "Car Part State Detection · EfficientNet-B0 · 2026<br>"
    "Dibuat oleh Rio Ramadhani Harllambang (1202220205) "
    "untuk Tugas Besar Mata Kuliah Pengantar Deep Learning"
    "</p>",
    unsafe_allow_html=True
)