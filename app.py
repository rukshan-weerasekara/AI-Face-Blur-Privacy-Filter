import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# --- Robust MediaPipe Import ---
try:
    import mediapipe as mp
    # Direct import for better stability in Streamlit Cloud
    from mediapipe.python.solutions import face_detection as mp_face_detection
except Exception as e:
    st.error(f"MediaPipe loading error: {e}")

# --- Page Configuration ---
st.set_page_config(page_title="AI Face Blur", layout="centered")

st.title("👤 AI Face Blur & Privacy Filter")
st.markdown("Developed by **Rukshan Weerasekara** | Creative Technologist")
st.markdown("---")

# Initialize Face Detection with caching to save memory
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
    # Load image and convert to RGB for MediaPipe
    img = Image.open(uploaded_file)
    img_array = np.array(img.convert('RGB'))
    h, w, _ = img_array.shape

    st.image(img, caption='Original Image', use_container_width=True)
    
    # Slider for Blur Intensity (Sigma)
    blur_strength = st.slider("Select Blur Strength", 5, 95, 35, step=2)

    if st.button("Apply AI Face Blur"):
        with st.spinner("AI is analyzing faces..."):
            # Step 1: Process the image
            results = face_detector.process(img_array)

            if results.detections:
                processed_img = img_array.copy()
                
                for detection in results.detections:
                    # --- FIXED: Use xmin and ymin instead of left/top ---
                    bbox = detection.location_data.relative_bounding_box
                    
                    # Convert relative coordinates (0 to 1) to actual pixel values
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    bw = int(bbox.width * w)
                    bh = int(bbox.height * h)

                    # Step 2: Boundary Safety Fixes
                    x, y = max(0, x), max(0, y)
                    bw = min(bw, w - x)
                    bh = min(bh, h - y)
                    
                    if bw > 0 and bh > 0:
                        # Step 3: Extract face region
                        face_region = processed_img[y:y+bh, x:x+bw]
                        
                        # Step 4: Apply Gaussian Blur
                        # Kernel size (k) must be an odd number
                        k = blur_strength if blur_strength % 2 != 0 else blur_strength + 1
                        blurred_face = cv2.GaussianBlur(face_region, (k, k), 0)
                        
                        # Step 5: Put blurred face back into image
                        processed_img[y:y+bh, x:x+bw] = blurred_face

                # Show Final Result
                st.image(processed_img, caption='Processed Image', use_container_width=True)
                
                # --- Step 6: Download Handling (Using BytesIO) ---
                result_pil = Image.fromarray(processed_img)
                buf = io.BytesIO()
                result_pil.save(buf, format="JPEG")
                
                st.download_button(
                    label="📥 Download Blurred Image",
                    data=buf.getvalue(),
                    file_name=f"blurred_{uploaded_file.name}",
                    mime="image/jpeg"
                )
                st.success(f"Detected {len(results.detections)} face(s)!")
            else:
                st.warning("No faces found. Try a clearer photo.")

st.markdown("---")
st.caption("AI Privacy Tool - Part of Ruka's Portfolio")
