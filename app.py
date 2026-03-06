import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Low Light Enhancement", layout="wide")

st.title("Low Light Image Enhancement (CLAHE)")

st.write("Upload a low-light image to enhance it using CLAHE.")

# Upload image
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

# Enhancement strength slider
clip_limit = st.slider("Enhancement Strength (CLAHE Clip Limit)", 1.0, 5.0, 2.0)

def enhance_clahe(image, clip_limit):
    
    img = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(img)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
    cl = clahe.apply(l)

    merged = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    return enhanced


if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    enhanced_img = enhance_clahe(img, clip_limit)

    st.subheader("Results")

    col1, col2 = st.columns(2)

    with col1:
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)

    with col2:
        st.image(cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB), caption="Enhanced", use_container_width=True)

    # Download button
    _, buffer = cv2.imencode(".png", enhanced_img)

    st.download_button(
        label="Download Enhanced Image",
        data=buffer.tobytes(),
        file_name="enhanced_image.png",
        mime="image/png"
    )