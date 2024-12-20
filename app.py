import streamlit as st
import cv2
import numpy as np
from PIL import Image as PILImage
from ultralytics import YOLO
from model import DetectNet

st.set_page_config(page_title="Crack Detection", page_icon=":mag:", layout="centered")

st.title("Crack Detection using YOLO V8")

# Load YOLO model
yolo = YOLO("best.pt")
save_dir = "detect.png"
model = DetectNet(yolo, save_name=save_dir)

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Convert the uploaded file to an OpenCV image
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Run detection
    result = model(img)
    
    # Hiển thị ảnh sau khi detect
    detected_img = cv2.imread(save_dir)
    # Chuyển BGR sang RGB để hiển thị trên Streamlit
    detected_img_rgb = cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB)
    
    st.image(detected_img_rgb, caption="Detected Image")

    # Xuất kết quả
    if len(result) == 0:
        st.write("No crack found")
    elif len(result) == 1:
        area, score = result[0]
        st.write(
            f"Crack predicted accuracy: {score:.2f}%\n"
            f"The area of crack is: {area:.2f} cm²"
        )
    else:
        for i, out in enumerate(result):
            area, score = out
            st.write(
                f"Crack {i+1} predicted accuracy: {score:.2f}%\n"
                f"The area of crack {i+1} is: {area:.2f} cm²\n"
            )
