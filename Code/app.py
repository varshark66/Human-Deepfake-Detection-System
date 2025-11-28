import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Page settings
st.set_page_config(page_title="DeepFake Detector", page_icon="🔍", layout="centered")

# ---------- Custom UI Theme (Peach + Pink) ----------
st.markdown("""
    <style>
        body {
            background-color: #FFE5E5;
        }
        .main {
            background: linear-gradient(135deg, #FFD1DC, #FFE6CC);
            padding: 2rem;
            border-radius: 20px;
        }
        .stButton>button {
            background-color: #FF8FA6;
            color: white;
            border-radius: 12px;
            font-size: 18px;
            font-weight: bold;
            padding: 0.6rem 1.4rem;
            border: none;
        }
        .stButton>button:hover {
            background-color: #FF6787;
            color: white;
        }
        .result-box {
            padding: 15px;
            border-radius: 12px;
            text-align: center;
            font-weight: bold;
            font-size: 22px;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align:center;color:#FF577F;'>DeepFake Image Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#FF6F91;'>Upload an image & verify authenticity within seconds 🔍</p>", unsafe_allow_html=True)

# ---------- Load model (safe) ----------
@st.cache_resource
def load_deepfake_model():
    try:
        return load_model('deepfake_detection_model.h5')
    except Exception as e:
        st.error(f"Model load failed: {e}")
        return None

model = load_deepfake_model()

# ---------- Preprocessing ----------
def preprocess_image(image_bgr):
    img = cv2.resize(image_bgr, (96, 96))
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

# ---------- Prediction ----------
THRESH = 0.50

def predict_image(image_bgr):
    x = preprocess_image(image_bgr)
    prob_real = float(model.predict(x, verbose=0)[0][0])
    label = "Real ✅" if prob_real >= THRESH else "Fake ❌"
    return label, prob_real

# ---------- Upload Section ----------
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(image_bgr, channels="BGR", caption="Uploaded Image", use_column_width=True)

    if model is None:
        st.stop()

    if st.button("Analyze 🔍"):
        with st.spinner("Analyzing, please wait... ⏳"):
            label, prob_real = predict_image(image_bgr)

        # 🔹 Color-coded results
        color = "#4BB543" if "Real" in label else "#FF4B4B"

        st.markdown(
            f"""
            <div class='result-box' style='background-color:{color}; color:white;'>
                {label}<br>
                Confidence: {prob_real:.2f}
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown("---")
st.caption("This is a demo classifier. Results may vary on unseen manipulations.")