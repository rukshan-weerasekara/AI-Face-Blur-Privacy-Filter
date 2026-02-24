# 👤 AI Face Blur & Privacy Filter

An intelligent, real-time image privacy tool that automatically detects human faces and applies a customizable Gaussian blur. Built for journalists, researchers, and creators who need to protect identities in visual content.

## 🔗 Live Demo
Experience the AI in action here: [https://ai-face-blur-privacy-filter-fohbwgxspywoewxhwzetcn.streamlit.app/](https://ai-face-blur-privacy-filter-fohbwgxspywoewxhwzetcn.streamlit.app/)

## 🚀 Why This Project?
In an era of digital surveillance and data privacy concerns, protecting individual identities in media is crucial. As a **Creative Technologist and Animator**, I developed this tool to provide a fast, AI-driven solution for privacy filtering without needing manual editing in software like After Effects or Photoshop.

### Key Features:
* **Automated Face Detection:** Powered by **Google MediaPipe**, detecting faces with high precision even in challenging angles.
* **Adjustable Privacy Levels:** Real-time slider to control the Gaussian blur intensity (Sigma).
* **High-Speed Processing:** Optimized for instant results using "Headless" computer vision processing.
* **Secure & Private:** No images are stored on the server; processing happens on-the-fly.
* **Instant Export:** Download the blurred result as a high-quality JPEG.



## 🛠️ Technical Breakdown
This application leverages professional computer vision techniques:

1.  **Detection Layer:** Uses the **MediaPipe Face Detection** model, which utilizes a BlazeFace sub-model for fast and accurate landmark localization.
2.  **Blurring Algorithm:** Implements a **Gaussian Blur** filter. Mathematically, it applies a Gaussian function to each pixel:
    $$G(x, y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2+y^2}{2\sigma^2}}$$
3.  **Coordinate Mapping:** Dynamically converts relative AI coordinates ($xmin, ymin$) to absolute image pixels to ensure perfectly aligned privacy masks.



## 💻 Tech Stack
* **Python 3.11**
* **MediaPipe** (AI Core)
* **OpenCV-Headless** (Computer Vision)
* **Streamlit** (UI Framework)
* **NumPy & Pillow** (Data Handling)

## 📦 Installation
To run this locally, clone the repo and install the requirements:
```bash
git clone [https://github.com/your-username/ai-face-blur-privacy-filter.git](https://github.com/your-username/ai-face-blur-privacy-filter.git)
pip install -r requirements.txt
streamlit run app.py
