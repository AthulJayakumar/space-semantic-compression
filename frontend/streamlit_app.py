"""frontend.streamlit_app

Plain-English purpose: Streamlit dashboard used for interactive demonstrations.

This file is part of the public Space Semantic Compression research demo.
It is documented at module level so researchers, supervisors, and non-specialist
readers can understand where the file fits before reading implementation details.
"""

from __future__ import annotations

import os
from pathlib import Path

import requests
import streamlit as st
from PIL import Image

API_URL = os.getenv("COMPRESSAI_API_URL", "http://localhost:8000")

st.set_page_config(page_title="Space Semantic Compression", page_icon=" ", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background: #08111f; color: #e6edf5; }
    section[data-testid="stSidebar"] { background: #0d1829; }
    div[data-testid="stMetric"] { background: #111f33; border: 1px solid #26384f; padding: 12px; border-radius: 6px; }
    .block-container { padding-top: 1.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Space AI Semantic Compression")
st.caption("Mission telemetry dashboard for semantic token downlink experiments")

with st.sidebar:
    st.header("Link Budget")
    bandwidth_kbps = st.slider("Bandwidth kbps", 16.0, 2048.0, 256.0, step=16.0)
    latency_ms = st.slider("Latency ms", 0.0, 2500.0, 600.0, step=50.0)
    packet_loss_percent = st.slider("Packet loss %", 0.0, 25.0, 2.0, step=0.5)
    outage_probability = st.slider("Outage probability", 0.0, 0.5, 0.05, step=0.01)
    semantic_keep_ratio = st.slider("Semantic token keep ratio", 0.05, 1.0, 0.45, step=0.05)
    semantic_method = st.selectbox("Semantic analyzer", ["hybrid", "saliency", "segmentation", "object"])

uploaded = st.file_uploader("Satellite or aerial image", type=["png", "jpg", "jpeg", "webp"])


def post_image(endpoint: str, extra: dict[str, str | float | int] | None = None) -> requests.Response:
    if uploaded is None:
        raise RuntimeError("No upload")
    uploaded.seek(0)
    files = {"image": (uploaded.name, uploaded.getvalue(), uploaded.type)}
    return requests.post(f"{API_URL}{endpoint}", files=files, data=extra or {}, timeout=240)


if uploaded:
    original = Image.open(uploaded).convert("RGB")
    overview, regions_tab, transmission_tab, benchmark_tab = st.tabs(
        ["Compression", "Semantic Regions", "Transmission", "Benchmark"]
    )

    form_data = {
        "bandwidth_kbps": bandwidth_kbps,
        "latency_ms": latency_ms,
        "packet_loss_percent": packet_loss_percent,
        "outage_probability": outage_probability,
        "semantic_keep_ratio": semantic_keep_ratio,
        "semantic_method": semantic_method,
    }

    with overview:
        with st.spinner("Encoding semantic tokens and reconstructing downlink image..."):
            response = post_image("/compress", form_data)

        if response.ok:
            result = response.json()
            left, middle, right = st.columns(3)
            with left:
                st.subheader("Original")
                st.image(original, use_container_width=True)
            with middle:
                st.subheader("Semantic Attention")
                heatmap_path = Path(result.get("semantic_heatmap_path") or "")
                if heatmap_path.exists():
                    st.image(str(heatmap_path), use_container_width=True)
            with right:
                st.subheader("Reconstruction")
                reconstruction_path = Path(result["reconstructed_image_path"])
                if reconstruction_path.exists():
                    st.image(str(reconstruction_path), use_container_width=True)

            metrics = st.columns(6)
            metrics[0].metric("Ratio", f"{result['compression_ratio']:.2f}x")
            metrics[1].metric("Saved", f"{result['bandwidth_saved_percent']:.1f}%")
            metrics[2].metric("PSNR", f"{result['psnr']:.2f} dB")
            metrics[3].metric("SSIM", f"{result['ssim']:.3f}")
            metrics[4].metric("Tokens", f"{result['semantic_token_count']}/{result['total_token_count']}")
            metrics[5].metric("Latency", f"{result['inference_latency_ms']:.1f} ms")

            st.subheader("Token Priority Mask")
            mask_path = Path(result.get("token_mask_path") or "")
            if mask_path.exists():
                st.image(str(mask_path), width=320)

            with st.expander("Full compression payload", expanded=False):
                st.json(result)
        else:
            st.error(response.json().get("detail", "Compression failed."))

    with regions_tab:
        if st.button("Analyze Semantic Regions", use_container_width=True):
            with st.spinner("Running semantic region analysis..."):
                response = post_image("/analyze-semantic-regions", {"method": semantic_method})
            if response.ok:
                analysis = response.json()
                st.metric("Detected regions", analysis["semantic_regions_detected"])
                st.metric("Semantic coverage", f"{analysis['semantic_coverage_percent']:.1f}%")
                st.dataframe(analysis["regions"], use_container_width=True)
            else:
                st.error(response.json().get("detail", "Semantic analysis failed."))

    with transmission_tab:
        if st.button("Simulate Satellite Downlink", use_container_width=True):
            with st.spinner("Simulating packetized semantic transmission..."):
                response = post_image("/simulate-transmission", form_data)
            if response.ok:
                result = response.json()
                transmission = result["transmission"]
                cols = st.columns(5)
                cols[0].metric("Full payload", f"{transmission['full_payload_kb']:.2f} KB")
                cols[1].metric("Semantic payload", f"{transmission['semantic_payload_kb']:.2f} KB")
                cols[2].metric("Time saved", f"{transmission['transmission_time_saved_sec']:.2f} s")
                cols[3].metric("Delivered packets", f"{transmission['packets_delivered']}/{transmission['packets_total']}")
                cols[4].metric("Semantic fidelity", f"{transmission['semantic_fidelity_percent']:.1f}%")
                st.json(transmission)
            else:
                st.error(response.json().get("detail", "Transmission simulation failed."))

    with benchmark_tab:
        if st.button("Run Research Benchmark", use_container_width=True):
            with st.spinner("Running VQ-VAE semantic benchmark suite..."):
                response = post_image("/benchmark")
            if response.ok:
                benchmark = response.json()
                st.success(f"Benchmark saved: {benchmark['benchmark_id']}")
                st.dataframe(benchmark["entries"], use_container_width=True)
                st.code(benchmark["csv_path"])
                st.code(benchmark["json_path"])
            else:
                st.error(response.json().get("detail", "Benchmark failed."))
