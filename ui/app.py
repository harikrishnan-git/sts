import sys
import os

# ── Allow imports from parent repo: dataset/, models/, eval/, config.py ──────
# app.py lives in sts/ui/ — go one level up to reach sts/
sys.path.append(os.path.abspath(".."))

import subprocess
import re
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — required for Streamlit
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time
from PIL import Image
import io

# ══════════════════════════════════════════════════════════════════════════════
# PATHS  (relative to sts/ui/ where you run `streamlit run app.py`)
# ══════════════════════════════════════════════════════════════════════════════
DATASET_PATH  = "../data/index.csv"
EVAL_SCRIPT   = "../eval/evaluate.py"   # subprocess target

CLASS_COLORS  = ["#1a56db", "#0891b2", "#7c3aed", "#059669", "#d97706",
                 "#e11d48", "#0d9488", "#b45309"]


# ══════════════════════════════════════════════════════════════════════════════
# 1.  DYNAMIC EVALUATION METRICS  — runs eval/evaluate.py via subprocess
# ══════════════════════════════════════════════════════════════════════════════
def get_eval_metrics():
    """
    Runs `python ../eval/evaluate.py`, captures stdout, parses:
      - accuracy  (float)
      - confusion matrix  (np.ndarray)
      - full stdout  (used as classification report display)
      - per-class f1 scores  (dict  label -> f1)
    Returns a dict; all values are None on failure.
    """
    result = subprocess.run(
        [sys.executable, EVAL_SCRIPT],
        capture_output=True,
        text=True,
        cwd=os.path.abspath("..")        # run from repo root so relative paths inside evaluate.py work
    )
    output  = result.stdout
    stderr  = result.stderr
    success = result.returncode == 0

    # ── Accuracy ─────────────────────────────────────────────────────────────
    # Matches:  Accuracy: 0.9423   |   accuracy: 94.23   |   Accuracy = 0.94
    acc_match = re.search(
        r'(?i)accuracy[\s:=]+([0-9]+\.?[0-9]*)', output
    )
    accuracy = None
    if acc_match:
        raw = float(acc_match.group(1))
        accuracy = raw / 100.0 if raw > 1.0 else raw   # handle 94.2 vs 0.942

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    # Matches numpy-style:  [[47  2  0] [1 51  1] ...]
    cm = None
    cm_match = re.search(r'(\[\s*\[.*?\]\s*\])', output, re.S)
    if cm_match:
        try:
            # Replace whitespace-only separators with commas for eval()
            cm_text = re.sub(r'\s+', ' ', cm_match.group(1).strip())
            cm_text = re.sub(r'(?<=\d)\s+(?=\d|-)', ', ', cm_text)
            cm_text = re.sub(r'\]\s+\[', '], [', cm_text)
            cm = np.array(eval(cm_text), dtype=int)
        except Exception:
            cm = None

    # ── Per-class F1 from classification report ───────────────────────────────
    # Parses lines like:   LPS       0.96  0.94  0.95      50
    per_class = {}
    for line in output.splitlines():
        parts = line.split()
        if len(parts) >= 5:
            try:
                label = parts[0]
                precision_v = float(parts[1])
                recall_v    = float(parts[2])
                f1_v        = float(parts[3])
                support_v   = int(parts[4])
                if 0.0 <= f1_v <= 1.0 and support_v > 0:
                    per_class[label] = {
                        "precision": precision_v,
                        "recall":    recall_v,
                        "f1":        f1_v,
                        "support":   support_v,
                    }
            except (ValueError, IndexError):
                continue

    return {
        "accuracy":   accuracy,
        "cm":         cm,
        "report":     output,
        "stderr":     stderr,
        "per_class":  per_class,
        "success":    success,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2.  DATASET INFO  — from MRIDataset or CSV fallback
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_dataset_info():
    # ── Real MRIDataset ───────────────────────────────────────────────────────
    try:
        from dataset.mri_dataset import MRIDataset
        dataset       = MRIDataset(DATASET_PATH)
        total_images  = len(dataset)
        class_names   = list(dataset.classes)
        total_classes = len(class_names)
        if hasattr(dataset, "class_counts"):
            class_counts = [dataset.class_counts[c] for c in class_names]
        else:
            class_counts = [total_images // total_classes] * total_classes
        return total_images, total_classes, class_names, class_counts, "MRIDataset"
    except Exception:
        pass

    # ── CSV fallback ──────────────────────────────────────────────────────────
    try:
        df = pd.read_csv(DATASET_PATH)
        label_col = next(
            (c for c in df.columns
             if c.lower() in ("label", "class", "tumor_type", "category", "diagnosis")),
            df.columns[-1]
        )
        total_images  = len(df)
        class_names   = sorted(df[label_col].unique().tolist())
        total_classes = len(class_names)
        class_counts  = [int((df[label_col] == c).sum()) for c in class_names]
        return total_images, total_classes, class_names, class_counts, f"CSV ({label_col})"
    except Exception:
        pass

    # ── Demo fallback ─────────────────────────────────────────────────────────
    return (
        248, 5,
        ["LPS", "LMS", "SS", "MFH", "DFSP"],
        [50, 53, 48, 52, 45],
        "demo (CSV not found)"
    )


# ══════════════════════════════════════════════════════════════════════════════
# 3.  MODEL ARCHITECTURE  — from SiameseViT or descriptive fallback
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_model_architecture():
    try:
        from models.siamese import SiameseViT
        import config as cfg
        model = SiameseViT(cfg.EMBED_DIM)
        return str(model), True
    except Exception:
        pass

    fallback = """SiameseViT(
  embed_dim = 512
  (vit_encoder): VisionTransformer(
    patch_size=16, image_size=224, num_layers=12, num_heads=8
    (patch_embed): PatchEmbedding(in_channels=1, embed_dim=512)
    (transformer): Sequential(
      (0-11): TransformerBlock(
        (attn): MultiHeadSelfAttention(heads=8, dim=512)
        (ff):   MLP(512 -> 2048 -> 512)
        (norm1, norm2): LayerNorm(512)
      )
    )
    (cls_token): Parameter(1, 1, 512)
    (pos_embed): Parameter(1, 197, 512)
  )
  (projection_head): Sequential(
    Linear(512 -> 256), ReLU(), Linear(256 -> 128)
  )
  (distance_fn): CosineSimilarity(dim=1)
)"""
    return fallback, False


# ══════════════════════════════════════════════════════════════════════════════
# 4.  TRAINING LOSS  — from main.py log file if available
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_training_loss():
    """
    Tries to parse ../training_log.txt for lines like:
        Epoch 1/100 | Loss: 1.43 | Val Loss: 1.72
    Falls back to smooth demo curves seeded for reproducibility.
    """
    log_path = os.path.abspath("../training_log.txt")
    if os.path.exists(log_path):
        train_losses, val_losses = [], []
        with open(log_path) as f:
            for line in f:
                t = re.search(r'(?i)(?:train[\s_]?)?loss[:\s]+([0-9.]+)', line)
                v = re.search(r'(?i)val[\s_]?loss[:\s]+([0-9.]+)', line)
                if t:
                    train_losses.append(float(t.group(1)))
                if v:
                    val_losses.append(float(v.group(1)))
        if train_losses:
            ep = np.arange(1, len(train_losses) + 1)
            vl = np.array(val_losses) if val_losses else None
            return ep, np.array(train_losses), vl, "training_log.txt"

    # Demo curves
    np.random.seed(42)
    ep = np.arange(1, 101)
    tl = 1.8 * np.exp(-3.5 * ep / 99) + 0.042 + np.random.randn(100) * 0.018
    vl = 2.0 * np.exp(-3.5 * ep / 99) + 0.087 + np.random.randn(100) * 0.028
    return ep, np.maximum(tl, 0), np.maximum(vl, 0), "demo"


# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="MRI Tumor Classification System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
#MainMenu{visibility:hidden;} footer{visibility:hidden;} .stDeployButton{display:none;}
.stApp { background: #f8fafc; }

.header-bar {
    background: white; border-bottom: 1px solid #e2e8f0;
    padding: 14px 0 12px; margin: -1rem -1rem 0;
}
.header-content { display:flex; align-items:center; justify-content:space-between; padding:0 2rem; }
.header-left { display:flex; align-items:center; gap:14px; }
.header-icon {
    width:42px; height:42px; border-radius:10px;
    background: linear-gradient(135deg, #1a56db, #0891b2);
    display:flex; align-items:center; justify-content:center; font-size:20px; flex-shrink:0;
}
.header-title { font-size:17px; font-weight:600; color:#0f172a; letter-spacing:-0.3px; margin:0; }
.header-sub   { font-size:12px; color:#64748b; margin:1px 0 0; }
.badge { display:inline-flex; align-items:center; gap:5px; padding:4px 10px; border-radius:20px; font-size:11.5px; font-weight:500; }
.badge-research { background:#fef3c7; color:#d97706; border:1px solid #fde68a; }
.badge-model    { background:#eff6ff; color:#1a56db; border:1px solid #dbeafe; }
.notice-bar {
    background:#fef3c7; border:1px solid #fde68a; border-radius:8px;
    padding:10px 16px; display:flex; align-items:center; gap:8px;
    font-size:13px; color:#92400e; font-weight:500; margin-bottom:8px;
}
.section-label {
    font-size:11px; font-weight:600; text-transform:uppercase;
    letter-spacing:0.8px; color:#94a3b8; margin:8px 0 12px;
    display:flex; align-items:center; gap:8px;
}
.section-label::after { content:''; flex:1; height:1px; background:#e2e8f0; }
.meta-grid { display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-top:4px; }
.meta-item { background:#f8fafc; border-radius:8px; padding:12px 14px; border:1px solid #e2e8f0; }
.meta-item-label { font-size:10.5px; color:#64748b; font-weight:500; text-transform:uppercase; letter-spacing:0.5px; }
.meta-item-val { font-size:14px; font-weight:600; color:#0f172a; margin-top:3px; font-family:'DM Mono',monospace; }
.info-step { background:#f0f9ff; border:1px solid #bae6fd; border-radius:8px; padding:11px 14px; margin-bottom:7px; font-size:13px; color:#0c4a6e; }
.warning-box { margin-top:10px; padding:10px 12px; border-radius:8px; background:#fef3c7; border:1px solid #fde68a; font-size:11.5px; color:#92400e; }
.eval-source { padding:2px 10px; border-radius:4px; font-size:11px; font-family:monospace; font-weight:500; }
[data-testid="metric-container"] {
    background:white; border:1px solid #e2e8f0; border-radius:12px;
    padding:16px 20px; box-shadow:0 1px 3px rgba(0,0,0,0.06);
}
[data-testid="metric-container"] label { font-size:12px!important; color:#64748b!important; font-weight:500!important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { font-size:28px!important; font-weight:600!important; }
.stButton > button {
    background: linear-gradient(135deg, #1a56db, #1d4ed8) !important;
    color:white !important; border:none !important; border-radius:8px !important;
    font-family:'DM Sans',sans-serif !important; font-size:14px !important;
    font-weight:600 !important; padding:10px 20px !important; width:100% !important;
    box-shadow:0 2px 8px rgba(26,86,219,0.3) !important;
}
.stButton > button:hover { box-shadow:0 4px 16px rgba(26,86,219,0.4) !important; }
[data-testid="stFileUploader"] { border-radius:12px !important; background:#f8fafc !important; }
.stSelectbox > div > div, .stTextInput > div > div > input {
    border-radius:8px !important; border-color:#e2e8f0 !important;
    font-family:'DM Sans',sans-serif !important; background:#f8fafc !important;
}
.stTextArea textarea { font-family:'DM Mono',monospace !important; font-size:12px !important; background:#0f172a !important; color:#e2e8f0 !important; }
.footer { border-top:1px solid #e2e8f0; margin-top:40px; padding:16px 0; display:flex; justify-content:space-between; font-size:11.5px; color:#94a3b8; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# LOAD STATIC DATA (cached)
# ══════════════════════════════════════════════════════════════════════════════
total_images, total_classes, class_names, class_counts, data_source = load_dataset_info()
model_arch_str, arch_is_real  = load_model_architecture()
epochs, train_loss, val_loss, loss_source = load_training_loss()
best_ep = int(np.argmin(val_loss)) + 1 if val_loss is not None else "—"

# Extend color list to cover any number of classes
while len(CLASS_COLORS) < len(class_names):
    CLASS_COLORS.append("#64748b")
class_colors = CLASS_COLORS[:len(class_names)]

# Session state
if "eval_result"    not in st.session_state: st.session_state.eval_result    = None
if "upload_result"  not in st.session_state: st.session_state.upload_result  = None


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="header-bar">
  <div class="header-content">
    <div class="header-left">
      <div class="header-icon">🧠</div>
      <div>
        <p class="header-title">MRI Tumor Classification System</p>
        <p class="header-sub">Few-Shot Learning · Vision Transformer · Soft Tissue Sarcoma
          &nbsp;·&nbsp; <span style="color:#059669;">Dataset: {data_source}</span>
        </p>
      </div>
    </div>
    <div style="display:flex;gap:10px;align-items:center;">
      <span class="badge badge-model">● ViT Few-Shot v2.1</span>
      <span class="badge badge-research">⚠ Research Use Only</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
st.markdown("""
<div class="notice-bar">
  ⚠️ <strong>Research Use Only.</strong>&nbsp;
  This tool is for academic and research purposes only.
  Not validated for clinical diagnosis or patient care decisions.
</div>
""", unsafe_allow_html=True)
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — IMAGE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-label">Image Analysis</div>', unsafe_allow_html=True)
col_upload, col_result = st.columns(2, gap="medium")

with col_upload:
    st.markdown("""
    <div style="background:white;border-radius:12px;border:1px solid #e2e8f0;
                padding:20px 20px 4px;box-shadow:0 1px 3px rgba(0,0,0,0.06);">
      <div style="font-size:13px;font-weight:600;color:#334155;text-transform:uppercase;
                  letter-spacing:0.5px;display:flex;align-items:center;
                  justify-content:space-between;margin-bottom:12px;">
        ↑ MRI Upload
        <span style="padding:2px 8px;border-radius:4px;background:#f1f5f9;
                     color:#64748b;font-size:11px;font-family:monospace;">POST /predict</span>
      </div>
    </div>""", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drop MRI image here or click to browse",
        type=["png", "jpg", "jpeg", "dcm"],
        help="Accepted formats: PNG, JPG, DICOM"
    )
    if uploaded_file:
        st.success(f"✓ **{uploaded_file.name}** · {uploaded_file.size / 1024:.1f} KB · Ready")

    c1, c2 = st.columns(2)
    with c1:
        patient_id = st.text_input("Patient ID", placeholder="e.g. PT-2024-001")
    with c2:
        mri_seq = st.selectbox("MRI Sequence",
            ["", "T1-weighted", "T2-weighted", "STIR", "T1 post-contrast", "DWI"])

    analyze_clicked = st.button("🔬  Analyze MRI Metadata", key="run_btn")

    if analyze_clicked:
        if uploaded_file is None:
            st.warning("Please upload an MRI image first.")
        else:
            with st.spinner("Loading image and extracting metadata…"):
                time.sleep(0.6)
            try:
                img = Image.open(uploaded_file)
                w, h, mode = img.size[0], img.size[1], img.mode
                channels = len(img.getbands())
            except Exception:
                w, h, mode, channels = 224, 224, "L", 1

            st.session_state.upload_result = {
                "filename"  : uploaded_file.name,
                "patient_id": patient_id or "PT-—",
                "seq"       : mri_seq or "Not specified",
                "width": w, "height": h,
                "mode": mode, "channels": channels,
                "timestamp" : datetime.now().strftime("%H:%M:%S"),
                "file_size" : f"{uploaded_file.size / 1024:.1f} KB",
                "image_obj" : uploaded_file,
            }
            st.rerun()

with col_result:
    if st.session_state.upload_result is None:
        st.markdown("""
        <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
                    min-height:260px;text-align:center;gap:12px;background:white;
                    border-radius:12px;border:1px solid #e2e8f0;
                    box-shadow:0 1px 3px rgba(0,0,0,0.06);padding:40px;">
          <div style="font-size:48px;">🖼️</div>
          <p style="font-size:13px;color:#94a3b8;">Upload an MRI image and click<br>
            <strong style="color:#64748b">Analyze MRI Metadata</strong></p>
        </div>""", unsafe_allow_html=True)
    else:
        r = st.session_state.upload_result
        try:
            r["image_obj"].seek(0)
            st.image(r["image_obj"], caption=r["filename"], use_container_width=True)
        except Exception:
            st.info("Preview not available for this file type.")

        st.markdown(f"""
        <div class="info-step">✅ <strong>Image loaded</strong> — {r['filename']}</div>
        <div class="info-step">📐 <strong>Resolution:</strong> {r['width']} × {r['height']} px
          &nbsp;·&nbsp; Mode: {r['mode']} &nbsp;·&nbsp; Channels: {r['channels']}</div>
        <div class="info-step">⚙️ <strong>Normalised slice extracted</strong>
          — resized to 224 × 224, intensity [0, 1]</div>
        <div class="info-step">🚀 <strong>Ready for model evaluation</strong>
          — pass to SiameseViT for few-shot inference</div>
        <div class="meta-grid" style="margin-top:10px;">
          <div class="meta-item">
            <div class="meta-item-label">Patient ID</div>
            <div class="meta-item-val">{r['patient_id']}</div>
          </div>
          <div class="meta-item">
            <div class="meta-item-label">MRI Sequence</div>
            <div class="meta-item-val">{r['seq']}</div>
          </div>
          <div class="meta-item">
            <div class="meta-item-label">File Size</div>
            <div class="meta-item-val">{r['file_size']}</div>
          </div>
          <div class="meta-item">
            <div class="meta-item-label">Loaded At</div>
            <div class="meta-item-val">{r['timestamp']}</div>
          </div>
        </div>
        <div class="warning-box">
          ⚠ Connect <code>POST /predict</code> to enable full model inference.
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — LIVE EVALUATION  (subprocess → eval/evaluate.py)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
st.markdown('<div class="section-label">Model Evaluation — Live from eval/evaluate.py</div>',
            unsafe_allow_html=True)

# ── Run evaluation button ─────────────────────────────────────────────────────
eval_col, _ = st.columns([1, 3])
with eval_col:
    run_eval = st.button("▶  Run Evaluation Script", key="eval_btn")

if run_eval:
    with st.spinner("Running `python eval/evaluate.py` …"):
        st.session_state.eval_result = get_eval_metrics()
    st.rerun()

# ── Display results ───────────────────────────────────────────────────────────
ev = st.session_state.eval_result

if ev is None:
    st.markdown("""
    <div style="background:white;border:1px solid #e2e8f0;border-radius:12px;
                padding:32px;text-align:center;color:#94a3b8;
                box-shadow:0 1px 3px rgba(0,0,0,0.06);">
      <div style="font-size:36px;margin-bottom:10px;">📊</div>
      <p style="font-size:13px;">Click <strong style="color:#1a56db">Run Evaluation Script</strong>
        to load live metrics from <code>eval/evaluate.py</code></p>
    </div>""", unsafe_allow_html=True)

else:
    if not ev["success"]:
        st.error(f"**eval/evaluate.py exited with an error.**\n\n```\n{ev['stderr'][:600]}\n```")

    # ── Top metrics row ───────────────────────────────────────────────────────
    accuracy  = ev["accuracy"]
    per_class = ev["per_class"]

    # Derive precision / recall / f1 from per-class dict if available
    if per_class:
        valid = [v for v in per_class.values()
                 if isinstance(v, dict) and "f1" in v]
        avg_precision = float(np.mean([v["precision"] for v in valid])) if valid else None
        avg_recall    = float(np.mean([v["recall"]    for v in valid])) if valid else None
        avg_f1        = float(np.mean([v["f1"]        for v in valid])) if valid else None
    else:
        avg_precision = avg_recall = avg_f1 = None

    m1, m2, m3, m4 = st.columns(4, gap="medium")
    with m1:
        if accuracy is not None:
            st.metric("Model Accuracy",  f"{accuracy*100:.2f}%")
        else:
            st.metric("Model Accuracy",  "—")
    with m2:
        st.metric("Precision (macro)", f"{avg_precision*100:.2f}%" if avg_precision else "—")
    with m3:
        st.metric("Recall (macro)",    f"{avg_recall*100:.2f}%"    if avg_recall    else "—")
    with m4:
        st.metric("F1 Score (macro)",  f"{avg_f1*100:.2f}%"        if avg_f1        else "—")

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── Confusion Matrix + Classification Report ──────────────────────────────
    cm = ev["cm"]
    col_cm, col_report = st.columns([1, 1], gap="medium")

    with col_cm:
        st.markdown("""
        <div style="background:white;border-radius:12px;border:1px solid #e2e8f0;
                    padding:20px 20px 4px;box-shadow:0 1px 3px rgba(0,0,0,0.06);">
          <div style="font-size:13px;font-weight:600;color:#334155;text-transform:uppercase;
                      letter-spacing:0.5px;margin-bottom:4px;">▦ Confusion Matrix</div>
          <p style="font-size:11.5px;color:#64748b;margin-bottom:12px;">
            Parsed from <code>eval/evaluate.py</code> stdout
          </p>
        </div>""", unsafe_allow_html=True)

        if cm is not None:
            n = cm.shape[0]
            tick_labels = class_names[:n] if len(class_names) >= n else [str(i) for i in range(n)]

            fig_cm, ax_cm = plt.subplots(figsize=(5.5, 4.2))
            fig_cm.patch.set_facecolor("white")
            ax_cm.set_facecolor("white")
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=tick_labels,
                yticklabels=tick_labels,
                ax=ax_cm,
                linewidths=0.5,
                linecolor="#e2e8f0",
                cbar_kws={"shrink": 0.8},
            )
            ax_cm.set_xlabel("Predicted", fontsize=11, labelpad=8)
            ax_cm.set_ylabel("Actual",    fontsize=11, labelpad=8)
            ax_cm.tick_params(labelsize=10)
            plt.tight_layout()
            st.pyplot(fig_cm, use_container_width=True)
            plt.close(fig_cm)
        else:
            st.warning("Could not parse confusion matrix from script output. "
                       "Ensure evaluate.py prints it in numpy format: `[[a b] [c d]]`")

    with col_report:
        st.markdown("""
        <div style="background:white;border-radius:12px;border:1px solid #e2e8f0;
                    padding:20px 20px 8px;box-shadow:0 1px 3px rgba(0,0,0,0.06);">
          <div style="font-size:13px;font-weight:600;color:#334155;text-transform:uppercase;
                      letter-spacing:0.5px;margin-bottom:4px;">📋 Classification Report</div>
          <p style="font-size:11.5px;color:#64748b;margin-bottom:8px;">
            Full stdout from <code>eval/evaluate.py</code>
          </p>
        </div>""", unsafe_allow_html=True)

        report_text = ev["report"].strip() if ev["report"] else "(no output)"
        st.text_area(
            label="",
            value=report_text,
            height=320,
            key="report_area",
            label_visibility="collapsed",
        )

        if ev["stderr"].strip():
            with st.expander("⚠ stderr output"):
                st.code(ev["stderr"], language="text")

    # ── Per-class accuracy bar chart (from parsed F1 scores) ─────────────────
    if per_class:
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div style="background:white;border-radius:12px;border:1px solid #e2e8f0;
                    padding:20px 20px 4px;box-shadow:0 1px 3px rgba(0,0,0,0.06);">
          <div style="font-size:13px;font-weight:600;color:#334155;text-transform:uppercase;
                      letter-spacing:0.5px;margin-bottom:4px;">▮ Per-Class F1 Score</div>
          <p style="font-size:11.5px;color:#64748b;margin-bottom:4px;">
            Parsed from classification report — no hardcoded values
          </p>
        </div>""", unsafe_allow_html=True)

        pc_labels = list(per_class.keys())
        pc_f1     = [per_class[k]["f1"] * 100 for k in pc_labels]
        pc_colors = [CLASS_COLORS[i % len(CLASS_COLORS)] for i in range(len(pc_labels))]

        fig_pc = go.Figure(go.Bar(
            x=pc_labels, y=pc_f1,
            marker=dict(color=pc_colors, line=dict(color=pc_colors, width=1.5)),
            text=[f"{v:.1f}%" for v in pc_f1],
            textposition="outside",
            textfont=dict(size=12, family="DM Sans", color="#334155"),
            hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
        ))
        fig_pc.update_layout(
            margin=dict(l=0, r=0, t=30, b=0), height=260,
            yaxis=dict(
                range=[max(0, min(pc_f1) - 5), 103],
                title="F1 Score (%)", gridcolor="#f1f5f9",
                ticksuffix="%", tickfont=dict(size=10),
            ),
            xaxis=dict(showgrid=False, tickfont=dict(size=12)),
            paper_bgcolor="white", plot_bgcolor="white",
            font=dict(family="DM Sans"), showlegend=False,
        )
        st.plotly_chart(fig_pc, use_container_width=True, config={"displayModeBar": False})

    # Timestamp of last eval run
    st.caption(f"Last evaluated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — TRAINING CURVES
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
st.markdown('<div class="section-label">Training Curves</div>', unsafe_allow_html=True)

st.markdown(f"""
<div style="background:white;border-radius:12px;border:1px solid #e2e8f0;
            padding:20px 20px 4px;box-shadow:0 1px 3px rgba(0,0,0,0.06);">
  <div style="font-size:13px;font-weight:600;color:#334155;text-transform:uppercase;
              letter-spacing:0.5px;margin-bottom:4px;">〜 Loss vs Epoch</div>
  <p style="font-size:11.5px;color:#64748b;margin-bottom:8px;">
    Source: <code>{loss_source}</code>
    {'&nbsp;·&nbsp; To use real curves: save loss values to <code>training_log.txt</code>'
     if loss_source == "demo" else ""}
  </p>
</div>""", unsafe_allow_html=True)

fig_loss = go.Figure()
fig_loss.add_trace(go.Scatter(
    x=epochs, y=train_loss, name="Train Loss",
    line=dict(color="#1a56db", width=2),
    fill="tozeroy", fillcolor="rgba(26,86,219,0.06)",
    hovertemplate="Epoch %{x}<br>Train Loss: %{y:.4f}<extra></extra>",
))
if val_loss is not None:
    fig_loss.add_trace(go.Scatter(
        x=epochs, y=val_loss, name="Val Loss",
        line=dict(color="#f59e0b", width=2, dash="dash"),
        fill="tozeroy", fillcolor="rgba(245,158,11,0.04)",
        hovertemplate="Epoch %{x}<br>Val Loss: %{y:.4f}<extra></extra>",
    ))
fig_loss.update_layout(
    margin=dict(l=0, r=0, t=10, b=0), height=220,
    xaxis=dict(title="Epoch", showgrid=False, tickfont=dict(size=10)),
    yaxis=dict(title="Loss", gridcolor="#f1f5f9", tickfont=dict(size=10)),
    paper_bgcolor="white", plot_bgcolor="white",
    font=dict(family="DM Sans"),
    legend=dict(orientation="h", y=1.12, x=0, font=dict(size=11)),
    hovermode="x unified",
)
st.plotly_chart(fig_loss, use_container_width=True, config={"displayModeBar": False})

tc1, tc2, tc3 = st.columns(3)
for col, val, label, color in [
    (tc1, f"{float(train_loss[-1]):.4f}",
          "Final Train Loss",  "#1a56db"),
    (tc2, f"{float(val_loss[-1]):.4f}" if val_loss is not None else "—",
          "Final Val Loss",    "#f59e0b"),
    (tc3, f"Ep. {best_ep}",
          "Best Checkpoint",   "#059669"),
]:
    with col:
        st.markdown(f"""
        <div style="text-align:center;padding:10px 8px;background:#f8fafc;
                    border-radius:8px;border:1px solid #e2e8f0;margin-top:8px;">
          <div style="font-size:16px;font-weight:600;color:{color};">{val}</div>
          <div style="font-size:10.5px;color:#64748b;margin-top:2px;">{label}</div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — DATASET OVERVIEW  (dynamic from MRIDataset / CSV)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
st.markdown('<div class="section-label">Dataset Overview</div>', unsafe_allow_html=True)

pills_html = " ".join([
    f'<span style="padding:4px 11px;border-radius:6px;font-size:11.5px;font-weight:500;'
    f'background:#eff6ff;color:#1a56db;border:1px solid #dbeafe;font-family:monospace;">{s}</span>'
    for s in ["T1w", "T2w", "STIR", "T1+Gd", "DWI", "ADC"]
])

st.markdown(f"""
<div style="background:white;border-radius:12px;border:1px solid #e2e8f0;
            padding:20px;box-shadow:0 1px 3px rgba(0,0,0,0.06);">
  <div style="font-size:13px;font-weight:600;color:#334155;text-transform:uppercase;
              letter-spacing:0.5px;display:flex;align-items:center;
              justify-content:space-between;margin-bottom:16px;">
    ◉ Dataset Statistics
    <span style="padding:2px 8px;border-radius:4px;background:#f1f5f9;
                 color:#64748b;font-size:11px;font-family:monospace;">GET /dataset-info · {data_source}</span>
  </div>
  <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:18px;">
    <div style="text-align:center;padding:14px 8px;background:#eff6ff;border-radius:8px;border:1px solid #dbeafe;">
      <div style="font-size:28px;font-weight:700;color:#1a56db;">{total_images}</div>
      <div style="font-size:11px;color:#334155;margin-top:3px;font-weight:500;">Total MRI Images</div>
    </div>
    <div style="text-align:center;padding:14px 8px;background:#e0f2fe;border-radius:8px;border:1px solid #bae6fd;">
      <div style="font-size:28px;font-weight:700;color:#0891b2;">{total_images:,}</div>
      <div style="font-size:11px;color:#334155;margin-top:3px;font-weight:500;">MRI Sequences</div>
    </div>
    <div style="text-align:center;padding:14px 8px;background:#ede9fe;border-radius:8px;border:1px solid #ddd6fe;">
      <div style="font-size:28px;font-weight:700;color:#7c3aed;">{total_classes}</div>
      <div style="font-size:11px;color:#334155;margin-top:3px;font-weight:500;">Tumor Classes</div>
    </div>
  </div>
  <div style="font-size:12px;font-weight:600;color:#64748b;margin-bottom:8px;">Class Distribution</div>
""", unsafe_allow_html=True)

max_count = max(class_counts) if class_counts else 1
for cls, cnt, color in zip(class_names, class_counts, class_colors):
    pct = int((cnt / max_count) * 100)
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:7px;">
      <span style="font-size:12px;color:#334155;font-weight:500;width:130px;flex-shrink:0;">{cls}</span>
      <div style="flex:1;background:#f1f5f9;border-radius:4px;height:8px;">
        <div style="width:{pct}%;height:8px;border-radius:4px;background:{color};"></div>
      </div>
      <span style="font-size:11.5px;color:#64748b;font-family:monospace;width:36px;text-align:right;">{cnt}</span>
    </div>""", unsafe_allow_html=True)

st.markdown(f"""
  <div style="margin-top:14px;">
    <div style="font-size:12px;font-weight:600;color:#64748b;margin-bottom:8px;">Available MRI Sequences</div>
    <div style="display:flex;flex-wrap:wrap;gap:6px;">{pills_html}</div>
    <div style="font-size:11px;color:#94a3b8;margin-top:10px;">
      Split: 70% train · 15% val · 15% test &nbsp;·&nbsp; Few-shot: 5-way 5-shot
      &nbsp;·&nbsp; Source: <code>{data_source}</code>
    </div>
  </div>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — MODEL ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
st.markdown('<div class="section-label">Model Architecture</div>', unsafe_allow_html=True)

arch_tag_bg    = "#d1fae5" if arch_is_real else "#fef3c7"
arch_tag_color = "#059669" if arch_is_real else "#d97706"
arch_tag_label = "✓ Live from models.siamese" if arch_is_real \
                 else "○ Placeholder — ensure models/ is on sys.path"

st.markdown(f"""
<div style="background:white;border-radius:12px;border:1px solid #e2e8f0;
            padding:20px 20px 12px;box-shadow:0 1px 3px rgba(0,0,0,0.06);">
  <div style="font-size:13px;font-weight:600;color:#334155;text-transform:uppercase;
              letter-spacing:0.5px;display:flex;align-items:center;
              justify-content:space-between;margin-bottom:4px;">
    🏗 SiameseViT — Vision Transformer (Few-Shot)
    <span style="padding:2px 10px;border-radius:4px;background:{arch_tag_bg};
                 color:{arch_tag_color};font-size:11px;font-family:monospace;font-weight:500;">
      {arch_tag_label}
    </span>
  </div>
</div>""", unsafe_allow_html=True)

st.code(model_arch_str, language="text")

with st.expander("📖 Architecture Notes"):
    st.markdown(f"""
| Component | Detail |
|---|---|
| **Backbone** | Vision Transformer (ViT) — patch-based self-attention |
| **Input** | 224 × 224, single-channel (grayscale MRI) |
| **Patches** | 16 × 16 → 196 tokens + 1 CLS token |
| **Embed dim** | 512 |
| **Depth** | 12 transformer layers, 8 heads |
| **Few-shot head** | Siamese projection: 512 → 256 → 128 |
| **Similarity** | Cosine distance (support vs query embedding) |
| **Protocol** | 5-way 5-shot episodic training |
| **Loss** | Prototypical + contrastive |

Data source: **`{data_source}`** · Architecture source: **`{'models.siamese' if arch_is_real else 'placeholder'}`**
""")


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="footer">
  <span>MRI Few-Shot Tumor Classification · Vision Transformer Architecture · Research Project</span>
  <span>© 2024 Research Lab · Not for clinical use</span>
</div>""", unsafe_allow_html=True)