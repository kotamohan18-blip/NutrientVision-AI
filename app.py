from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from PIL import Image

from src.inference import (
    CLASS_LABELS,
    MAX_UPLOAD_MB,
    load_uploaded_image,
    predict_images,
)
from src.model import load_densenet121

APP_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = APP_ROOT / "models" / "densenet121_best.pth"
MODEL_PATH = Path(os.environ.get("MODEL_PATH", str(DEFAULT_MODEL_PATH))).expanduser()

st.set_page_config(
    page_title="NutrientVision AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Keep the UI restrained and readable; all actual model outputs come from inference.
st.markdown(
    """
    <style>
      :root {
        --nv-bg: #09090B;
        --nv-panel: #18181B;
        --nv-panel-2: #202024;
        --nv-border: #303036;
        --nv-muted: #A1A1AA;
        --nv-text: #F4F4F5;
        --nv-accent: #D65A63;
        --nv-green: #86A995;
      }
      .stApp { background: var(--nv-bg); color: var(--nv-text); }
      [data-testid="stAppViewContainer"] { background: var(--nv-bg); }
      [data-testid="stHeader"] { background: rgba(9,9,11,0.92); }
      .block-container { max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem; }
      [data-testid="stSidebar"] { background: #111114; border-right: 1px solid var(--nv-border); }
      h1, h2, h3, h4, p, label { color: var(--nv-text); }
      .nv-brand { display:flex; align-items:center; gap:14px; margin: 4px 0 6px; }
      .nv-mark {
        width:48px;height:48px;border-radius:15px;display:flex;align-items:center;
        justify-content:center;background:linear-gradient(145deg,#25252B,#141418);
        border:1px solid #3B3B44;font-size:25px;
      }
      .nv-title { font-size:30px; font-weight:750; letter-spacing:-.8px; line-height:1.1; }
      .nv-subtitle { color:var(--nv-muted); font-size:13px; margin-top:5px; }
      .nv-pill {
        display:inline-flex;align-items:center;gap:6px;padding:5px 10px;border-radius:99px;
        border:1px solid #3E3E46;color:#C4C4CC;font-size:11px;background:#17171B;
      }
      .nv-section-label {
        color:#B9B9C2;font-size:11px;font-weight:700;letter-spacing:1.3px;
        text-transform:uppercase;margin-bottom:8px;
      }
      .nv-card {
        background:var(--nv-panel); border:1px solid var(--nv-border);
        border-radius:16px; padding:18px 20px;
      }
      .nv-muted { color:var(--nv-muted); font-size:13px; }
      .nv-result-label { color:var(--nv-muted);font-size:12px;letter-spacing:.3px; }
      .nv-result-value { color:var(--nv-text);font-size:27px;font-weight:750;margin-top:3px; }
      .nv-note {
        border-left:3px solid var(--nv-accent);background:#19191D;padding:11px 14px;
        border-radius:0 10px 10px 0;color:#C9C9D0;font-size:12px;line-height:1.55;
      }
      .nv-kpi {
        border:1px solid var(--nv-border);border-radius:13px;background:#151519;
        padding:13px 15px;min-height:83px;
      }
      .nv-kpi-label { color:var(--nv-muted);font-size:11px; }
      .nv-kpi-value { color:var(--nv-text);font-size:22px;font-weight:700;margin-top:5px; }
      [data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap:22px; border-bottom:1px solid var(--nv-border);
      }
      [data-testid="stTabs"] [data-baseweb="tab"] {
        height:48px; background:transparent; color:#A1A1AA; font-weight:600;
      }
      [data-testid="stTabs"] [aria-selected="true"] { color:#F4F4F5 !important; }
      div.stButton > button[kind="primary"] {
        background:var(--nv-accent);border:1px solid var(--nv-accent);
        border-radius:10px;color:white;font-weight:650;min-height:43px;
      }
      div.stButton > button[kind="primary"]:hover { background:#C74D57;border-color:#C74D57; }
      div.stButton > button {
        border-radius:10px;border-color:#42424A;color:#E4E4E7;
      }
      [data-testid="stFileUploader"] section {
        border:1px dashed #4B4B55;border-radius:14px;background:#141418;
      }
      [data-testid="stMetric"] {
        background:#151519;border:1px solid var(--nv-border);padding:12px 14px;
        border-radius:12px;
      }
      [data-testid="stDataFrame"] { border:1px solid var(--nv-border);border-radius:10px; }
      hr { border-color:var(--nv-border); }
      footer { visibility:hidden; }
      @media (max-width: 720px) {
        .nv-title { font-size:23px; }
        .block-container { padding-left:1rem;padding-right:1rem; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def pretty_label(label: str) -> str:
    return {
        "Vitamin_A": "Vitamin A",
        "Vitamin_C": "Vitamin C",
        "Vitamin_D": "Vitamin D",
    }.get(label, label)


@st.cache_resource(show_spinner=False)
def get_model_cached(model_path: str, modified_time: float, device_name: str):
    # modified_time invalidates the cache if the checkpoint is replaced.
    del modified_time
    device = torch.device(device_name)
    return load_densenet121(model_path, device=device)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def render_model_status() -> tuple[Any | None, torch.device, bool]:
    device = get_device()
    exists = MODEL_PATH.is_file()
    if not exists:
        st.warning(
            "The trained DenseNet121 weights are not installed yet. "
            "Add your notebook checkpoint at `models/densenet121_best.pth` "
            "or set the `MODEL_PATH` environment variable. Prediction buttons "
            "will remain disabled until the trained weights are available.",
            icon="⚠️",
        )
        return None, device, False

    try:
        stat = MODEL_PATH.stat()
        with st.spinner("Loading the trained DenseNet121 checkpoint..."):
            model = get_model_cached(str(MODEL_PATH), stat.st_mtime, str(device))
        return model, device, True
    except Exception as exc:
        st.error(
            f"The checkpoint could not be loaded: {exc}. "
            "Check that this is the six-class `densenet121_best.pth` state dictionary "
            "created by your project notebook."
        )
        return None, device, False


def score_chart(scores: dict[str, float], selected_label: str):
    labels = [pretty_label(k) for k in scores]
    values = [scores[k] * 100 for k in scores]
    colors = [
        "#D65A63" if k == selected_label else "#74818B"
        for k in scores
    ]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{v:.1f}%" for v in values],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(
        height=300,
        margin=dict(l=8, r=38, t=8, b=22),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E4E4E7", size=12),
        xaxis=dict(
            range=[0, max(100, max(values, default=0) + 10)],
            title="Model score (%)",
            gridcolor="#303036",
            zerolinecolor="#303036",
        ),
        yaxis=dict(
            autorange="reversed",
            gridcolor="rgba(0,0,0,0)",
        ),
        showlegend=False,
        bargap=0.34,
    )
    return fig


def show_single_result(filename: str, image: Image.Image, row: dict):
    label = row["predicted_label"]
    score = row["model_score"] * 100
    left, right = st.columns([0.82, 1.18], gap="large")
    with left:
        st.image(image, caption=filename, use_container_width=True)
    with right:
        st.markdown('<div class="nv-section-label">Prediction result</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="nv-card"><div class="nv-result-label">Predicted dataset label</div>'
            f'<div class="nv-result-value">{pretty_label(label)}</div>'
            f'<div class="nv-muted" style="margin-top:8px">Model score · {score:.1f}%</div></div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            score_chart(row["scores"], label),
            use_container_width=True,
            config={"displayModeBar": False},
        )
    st.markdown(
        '<div class="nv-note">These values are softmax scores from the image classifier. '
        'They are not calibrated medical probabilities and do not establish the presence '
        'or absence of a nutrient deficiency.</div>',
        unsafe_allow_html=True,
    )


# Header
st.markdown(
    """
    <div class="nv-brand">
      <div class="nv-mark">🧬</div>
      <div>
        <div class="nv-title">NutrientVision AI</div>
        <div class="nv-subtitle">Six-class nutrient-related image classification · DenseNet121</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")
model, device, model_ready = render_model_status()

tabs = st.tabs(["◈  Predict", "▦  Batch images", "▤  Model information"])

with tabs[0]:
    st.markdown("## Classify an image")
    st.caption("Upload one image to view the model's predicted dataset label and class scores.")
    uploaded = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
        key="single_image",
        help=f"Maximum file size: {MAX_UPLOAD_MB} MB. The image is resized to 224 × 224 pixels.",
    )

    if uploaded is not None:
        try:
            image = load_uploaded_image(uploaded.getvalue())
            st.markdown('<div class="nv-section-label">Image preview</div>', unsafe_allow_html=True)
            if st.button("Predict category", type="primary", disabled=not model_ready, use_container_width=False):
                with st.spinner("Running DenseNet121 inference..."):
                    rows = predict_images(model, [image], device=device, batch_size=1)
                show_single_result(uploaded.name, image, rows[0])
            else:
                st.image(image, caption=uploaded.name, width=360)
                if not model_ready:
                    st.info("Install the trained checkpoint to enable prediction.")
        except Exception as exc:
            st.error(f"Could not read this image: {exc}")

with tabs[1]:
    st.markdown("## Batch image classification")
    st.caption("Process several images with the same DenseNet121 checkpoint and export the results.")
    batch_files = st.file_uploader(
        "Choose multiple images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="batch_images",
        help=f"Up to 100 images. Each image must be under {MAX_UPLOAD_MB} MB.",
    )
    if batch_files:
        st.write(f"{len(batch_files)} file(s) selected.")
        if len(batch_files) > 100:
            st.error("Please select no more than 100 images per batch.")
        elif st.button("Process batch", type="primary", disabled=not model_ready):
            valid_images = []
            valid_names = []
            invalid_rows = []
            for item in batch_files:
                try:
                    valid_images.append(load_uploaded_image(item.getvalue()))
                    valid_names.append(item.name)
                except Exception as exc:
                    invalid_rows.append({"File": item.name, "Status": f"Invalid image: {exc}"})

            result_rows = []
            if valid_images:
                progress = st.progress(0, text="Preparing batch...")
                try:
                    predictions = predict_images(
                        model, valid_images, device=device, batch_size=16
                    )
                    for idx, (name, prediction) in enumerate(zip(valid_names, predictions), start=1):
                        result = {
                            "File": name,
                            "Predicted label": pretty_label(prediction["predicted_label"]),
                            "Model score (%)": round(prediction["model_score"] * 100, 2),
                            "Status": "Predicted",
                        }
                        for class_name, score in prediction["scores"].items():
                            result[f"{pretty_label(class_name)} score (%)"] = round(score * 100, 2)
                        result_rows.append(result)
                        progress.progress(idx / len(valid_images), text=f"Processed {idx}/{len(valid_images)}")
                    progress.empty()
                except Exception as exc:
                    st.error(f"Batch inference stopped: {exc}")

            combined = result_rows + invalid_rows
            if combined:
                df = pd.DataFrame(combined)
                st.dataframe(df, use_container_width=True, hide_index=True)
                if result_rows:
                    csv = pd.DataFrame(result_rows).to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        "Download results (CSV)",
                        data=csv,
                        file_name="nutrientvision_predictions.csv",
                        mime="text/csv",
                        type="primary",
                    )
            if invalid_rows:
                st.warning(f"{len(invalid_rows)} file(s) could not be processed.")
    st.markdown(
        '<div class="nv-note">Batch predictions use the same preprocessing and model as the single-image page. '
        'A dataset label is a model output, not a clinical finding.</div>',
        unsafe_allow_html=True,
    )

with tabs[2]:
    st.markdown("## Model information")
    st.caption("The deployed inference path is intentionally limited to the strongest standalone model from the project comparison.")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Architecture", "DenseNet121")
    with c2:
        st.metric("Held-out test accuracy", "90.91%")
    with c3:
        st.metric("Test images", "418")

    st.markdown("### Training and input configuration")
    info_left, info_right = st.columns(2)
    with info_left:
        st.markdown(
            """
            **Model**
            - Pretrained torchvision DenseNet121
            - Final classifier: 1,024 → 6 outputs
            - Best checkpoint: `densenet121_best.pth`
            - Checkpoint selected by validation accuracy
            - Validation accuracy recorded: 90.53%
            """
        )
    with info_right:
        st.markdown(
            """
            **Inference preprocessing**
            - Convert image to RGB
            - Resize to 224 × 224
            - Convert to tensor
            - ImageNet mean: 0.485, 0.456, 0.406
            - ImageNet std: 0.229, 0.224, 0.225
            """
        )

    st.markdown("### Dataset labels")
    label_cols = st.columns(3)
    for idx, label in enumerate(CLASS_LABELS):
        with label_cols[idx % 3]:
            st.markdown(
                f'<div class="nv-kpi"><div class="nv-kpi-label">CLASS {idx + 1}</div>'
                f'<div class="nv-kpi-value">{pretty_label(label)}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("### Dataset summary")
    st.dataframe(
        pd.DataFrame(
            [
                {"Split": "Training", "Images": 3287},
                {"Split": "Validation", "Images": 412},
                {"Split": "Test", "Images": 418},
                {"Split": "Total", "Images": 4117},
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.markdown(
        '<div class="nv-note">Research limitation: the supplied dataset documentation does not establish '
        'the image source or laboratory confirmation of the folder labels. Iron and Zinc are minerals. '
        'The reported 90.91% is the result on this project’s held-out test split; it is not evidence of '
        'clinical diagnostic accuracy. This application is an educational research demonstration only.</div>',
        unsafe_allow_html=True,
    )

st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#71717A;font-size:11px">'
    'NutrientVision AI · Research prototype · Not for medical diagnosis</div>',
    unsafe_allow_html=True,
)
