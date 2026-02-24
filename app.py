import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# --- Robust MediaPipe Import ---
try:
    import mediapipe as mp
    # Accessing the solution directly from the sub-module to avoid AttributeError
    from mediapipe.python.solutions import face_detection as mp_face_detection
except Exception as e:
    st.error(f"MediaPipe loading error: {e}")

# --- Page Configuration ---
st.set_page_config(page_title="AI Face Blur", layout="centered")

st.title("👤 AI Face Blur & Privacy Filter")
st.markdown("Developed by **Rukshan Weerasekara** | Creative Technologist")
st.markdown("---")

# Initialize Face Detection with caching
@st.cache_resource
def get_face_detector():
    return mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

try:
    face_detector = get_face_detector()
except Exception as e:
    st.error("Could not initialize AI model. Please check logs.")

# --- File Uploader ---
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    img_array = np.array(img.convert('RGB'))
    h, w, _ = img_array.shape

    st.image(img, caption='Original Image', use_container_width=True)
    
    # Slider for Blur Intensity (Sigma)
    # The Gaussian formula used is:
    # $$G(x, y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2+y^2}{2\sigma^2}}$$
    blur_strength = st.slider("Select Blur Strength", 5, 95, 35, step=2)

    if st.button("Apply AI Face Blur"):
        with st.spinner("AI is analyzing faces..."):
            # Process the image
            results = face_detector.process(img_array)

            if results.detections:
                processed_img = img_array.copy()
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x, y, bw, bh = int(bbox.left * w), int(bbox.top * h), int(bbox.width * w), int(bbox.height * h)

                    # Boundary fixes
                    x, y = max(0, x), max(0, y)
                    bw, bh = min(bw, w - x), min(bh, h - y)
                    
                    if bw > 0 and bh > 0:
                        face_region = processed_img[y:y+bh, x:x+bw]
                        # Ensure kernel size is odd
                        k = blur_strength if blur_strength % 2 != 0 else blur_strength + 1
                        blurred_face = cv2.GaussianBlur(face_region, (k, k), 0)
                        processed_img[y:y+bh, x:x+bw] = blurred_face

                st.image(processed_img, caption='Processed Image', use_container_width=True)
                
                # Download logic
                result_pil = Image.fromarray(processed_img)
                buf = io.BytesIO()
                result_pil.save(buf, format="JPEG")
                st.download_button(
                    label="📥 Download Result",
                    data=buf.getvalue(),
                    file_name=f"blurred_{uploaded_file.name}",
                    mime="image/jpeg"
                )
                st.success(f"Detected {len(results.detections)} face(s)!")
            else:
                st.warning("No faces found. Try a clearer photo.")



st.markdown("---")
st.caption("AI Privacy Tool - Part of Ruka's Portfolio")
