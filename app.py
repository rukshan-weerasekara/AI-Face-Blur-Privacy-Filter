import streamlit as st
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp


st.set_page_config(page_title="AI Face Blur", layout="centered")

st.title("👤 AI Face Blur & Privacy Filter")
st.markdown("Protect privacy by automatically blurring faces in your images using AI.")

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# File Uploader
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert uploaded file to OpenCV format
    img = Image.open(uploaded_file)
    img_array = np.array(img.convert('RGB'))
    h, w, _ = img_array.shape

    st.image(img, caption='Original Image', use_container_width=True)
    
    blur_strength = st.slider("Select Blur Strength", 5, 95, 35, step=2)

    if st.button("Apply AI Face Blur"):
        with st.spinner("AI is detecting faces..."):
            #Detect faces using MediaPipe
            results = face_detection.process(img_array)

            if results.detections:
                processed_img = img_array.copy()
                
                for detection in results.detections:
                    # Get bounding box coordinates
                    bboxC = detection.location_data.relative_bounding_box
                    x, y, bw, bh = int(bboxC.left * w), int(bboxC.top * h), int(bboxC.width * w), int(bboxC.height * h)

                    # Ensure coordinates are within image boundaries
                    x, y = max(0, x), max(0, y)
                    
                    # Extract the face region and apply Gaussian Blur
                    face_region = processed_img[y:y+bh, x:x+bw]
                    blurred_face = cv2.GaussianBlur(face_region, (blur_strength, blur_strength), 0)
                    
                    #Place the blurred face back onto the image
                    processed_img[y:y+bh, x:x+bw] = blurred_face

                st.image(processed_img, caption='Processed Image', use_container_width=True)
                st.success(f"Successfully blurred {len(results.detections)} face(s)!")
                
                # Option to download the result
                result_pil = Image.fromarray(processed_img)
                st.download_button(label="📥 Download Image", data=uploaded_file, file_name="blurred_image.jpg", mime="image/jpeg")
            else:
                st.warning("No faces detected in this image.")

st.markdown("---")
st.caption("Developed by Rukshan Weerasekara | Creative Technologist")
