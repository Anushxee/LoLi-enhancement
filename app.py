import streamlit as st
import cv2
import numpy as np

def enhance_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l2 = clahe.apply(l)

    merged = cv2.merge((l2,a,b))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    return enhanced


st.title("Low Light Image Enhancement (CLAHE)")

uploaded = st.file_uploader("Upload Image", type=["jpg","png"])

if uploaded:

    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    enhanced = enhance_clahe(img)

    st.subheader("Results")

    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="Original")

    with col2:
        st.image(enhanced, caption="Enhanced")