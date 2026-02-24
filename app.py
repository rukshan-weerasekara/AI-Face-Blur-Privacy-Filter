import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io


import mediapipe as mp
from mediapipe.python.solutions import face_detection as mp_face_detection


st.set_page_config(page_title="AI Face Blur", layout="centered")

st.title("👤 AI Face Blur & Privacy Filter")
st.markdown("Protect privacy by automatically blurring faces in your images using AI.")
st.markdown("Developed by **Rukshan Weerasekara** | Creative Technologist")
st.markdown("---")


face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)


uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load and display the original image
    img = Image.open(uploaded_file)
    img_array = np.array(img.convert('RGB'))
    h, w, _ = img_array.shape

    st.image(img, caption='Original Image', use_container_width=True)
    
  
    blur_strength = st.slider("Select Blur Strength", 5, 95, 35, step=2)

    if st.button("Apply AI Face Blur"):
        with st.spinner("AI is detecting faces..."):
            # Detect faces using MediaPipe's process method
            results = face_detection.process(img_array)

            if results.detections:
                processed_img = img_array.copy()
                
                for detection in results.detections:
                   
                    bboxC = detection.location_data.relative_bounding_box
                    x, y, bw, bh = int(bboxC.left * w), int(bboxC.top * h), int(bboxC.width * w), int(bboxC.height * h)

                   
                    x, y = max(0, x), max(0, y)
                    bw = min(bw, w - x)
                    bh = min(bh, h - y)
                    
                    #Extract face region and apply Gaussian Blur
                    face_region = processed_img[y:y+bh, x:x+bw]
                    # Note: GaussianBlur kernel size must be an odd number
                    k_size = blur_strength if blur_strength % 2 != 0 else blur_strength + 1
                    blurred_face = cv2.GaussianBlur(face_region, (k_size, k_size), 0)
                    
                    # Overlay the blurred region back onto the original image
                    processed_img[y:y+bh, x:x+bw] = blurred_face

                # Display the final result
                st.image(processed_img, caption='Processed Image', use_container_width=True)
                st.success(f"Successfully blurred {len(results.detections)} face(s)!")
                
                # Proper Image Download Handling ---
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
                st.warning("No faces detected in this image. Try another one.")

st.markdown("---")
st.caption("AI Privacy Tool - Build with MediaPipe and Streamlit")
