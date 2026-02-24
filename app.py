import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# --- Standard MediaPipe Import ---
try:
    import mediapipe as mp
    mp_face_detection = mp.solutions.face_detection
except ImportError as e:
    st.error(f"MediaPipe installation failed: {e}")

# --- Page Configuration ---
st.set_page_config(page_title="AI Face Blur", layout="centered")

st.title("👤 AI Face Blur & Privacy Filter")
st.markdown("Developed by **Rukshan Weerasekara** | Creative Technologist")
st.markdown("---")

# Initialize Face Detection
@st.cache_resource
def load_detector():
    return mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

try:
    face_detection = load_detector()
except NameError:
    st.error("Face detection could not be initialized. Check your requirements.txt")

# --- File Uploader ---
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    img_array = np.array(img.convert('RGB'))
    h, w, _ = img_array.shape

    st.image(img, caption='Original Image', use_container_width=True)
    
    blur_strength = st.slider("Select Blur Strength", 5, 95, 35, step=2)

    if st.button("Apply AI Face Blur"):
        with st.spinner("AI is analyzing faces..."):
            results = face_detection.process(img_array)

            if results.detections:
                processed_img = img_array.copy()
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    x, y, bw, bh = int(bboxC.left * w), int(bboxC.top * h), int(bboxC.width * w), int(bboxC.height * h)

                    # Safety boundary checks
                    x, y = max(0, x), max(0, y)
                    bw = min(bw, w - x)
                    bh = min(bh, h - y)
                    
                    if bw > 0 and bh > 0:
                        face_region = processed_img[y:y+bh, x:x+bw]
                        k_size = blur_strength if blur_strength % 2 != 0 else blur_strength + 1
                        blurred_face = cv2.GaussianBlur(face_region, (k_size, k_size), 0)
                        processed_img[y:y+bh, x:x+bw] = blurred_face

                st.image(processed_img, caption='Processed Image', use_container_width=True)
                
                # Image Download Handling
                result_pil = Image.fromarray(processed_img)
                buf = io.BytesIO()
                result_pil.save(buf, format="JPEG")
                byte_im = buf.getvalue()

                st.download_button(
                    label="📥 Download Blurred Image",
                    data=byte_im,
                    file_name=f"blurred_{uploaded_file.name}",
                    mime="image/jpeg"
                )
            else:
                st.warning("No faces detected.")

st.markdown("---")
st.caption("AI Privacy Tool by Ruka")
