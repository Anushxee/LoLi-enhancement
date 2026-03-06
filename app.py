import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_image_comparison import image_comparison

st.set_page_config(page_title="Low Light Enhancement", layout="wide")

st.title("Low Light Image Enhancement (CLAHE)")
st.write("Upload a low-light image and enhance it.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

clip_limit = st.slider("Enhancement Strength", 1.0, 5.0, 2.0)


def enhance_clahe(image, clip_limit):

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
    cl = clahe.apply(l)

    merged = cv2.merge((cl,a,b))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    return enhanced


if uploaded_file:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    enhanced = enhance_clahe(img, clip_limit)

    original_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)

    st.subheader("Interactive Comparison")

    image_comparison(
        img1=original_rgb,
        img2=enhanced_rgb,
        label1="Original",
        label2="Enhanced",
        width=700
    )

    st.subheader("Download Result")

    _, buffer = cv2.imencode(".png", enhanced)

    st.download_button(
        label="Download Enhanced Image",
        data=buffer.tobytes(),
        file_name="enhanced_image.png",
        mime="image/png"
    )