
import streamlit as st
import numpy as np
import cv2
import pandas as pd
from PIL import Image

st.set_page_config(
    page_title="TumorSpot AI",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 TumorSpot AI")
st.subheader("Tumor Microenvironment Explorer")

st.write(
    "Upload a tissue image to explore image regions "
    "and estimate their color-based distribution."
)

st.warning(
    "Research prototype only. This version uses color-based "
    "image processing, not trained AI. It cannot reliably "
    "identify tumor, stroma or immune cells."
)

uploaded = st.file_uploader(
    "Upload an H&E tissue image",
    type=["png", "jpg", "jpeg"]
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    image.thumbnail((1600, 1600))

    rgb = np.array(image)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    tissue = (s > 20) & (v < 245)

    # Color proxies, NOT biological classifications.
    purple = tissue & (h >= 115) & (h <= 165)
    pink = tissue & ((h <= 15) | (h >= 170))
    other = tissue & ~(purple | pink)

    masks = {
        "Purple color proxy": purple,
        "Pink color proxy": pink,
        "Other tissue": other
    }

    counts = {
        name: int(mask.sum())
        for name, mask in masks.items()
    }

    total = sum(counts.values())

    percentages = {
        name: (count / total * 100 if total else 0)
        for name, count in counts.items()
    }

    overlay = rgb.copy()
    colors = {
        "Purple color proxy": (70, 100, 255),
        "Pink color proxy": (40, 220, 100),
        "Other tissue": (255, 190, 0)
    }

    for name, mask in masks.items():
        overlay[mask] = (
            0.55 * rgb[mask]
            + 0.45 * np.array(colors[name])
        ).astype(np.uint8)

    overlay[~tissue] = rgb[~tissue]

    left, right = st.columns(2)

    with left:
        st.subheader("Original image")
        st.image(rgb, use_container_width=True)

    with right:
        st.subheader("Color-region overlay")
        st.image(overlay, use_container_width=True)

    st.subheader("Image measurements")

    st.metric("Image width", f"{image.width} pixels")
    st.metric("Image height", f"{image.height} pixels")
    st.metric("Detected tissue pixels", f"{total:,}")

    results = pd.DataFrame({
        "Region": list(percentages.keys()),
        "Area percentage": [
            round(x, 2) for x in percentages.values()
        ]
    })

    st.dataframe(results, hide_index=True)

    st.bar_chart(
        results.set_index("Region")["Area percentage"]
    )

    st.download_button(
        "Download results as CSV",
        data=results.to_csv(index=False),
        file_name="tumorspot_results.csv",
        mime="text/csv"
    )

    st.caption(
        "Color-region percentages are experimental image-processing "
        "measurements, not validated tumor, stroma or immune-cell "
        "percentages. Results are not for clinical use."
    )
else:
    st.info("Upload an image to begin.")
