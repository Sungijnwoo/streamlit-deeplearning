import streamlit as st
from PIL import Image
import torch
import os

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom

st.title('JINWOO DEEPLEARNING')


@st.cache
def load_image(image_file):
    img = Image.open(image_file)
    return img

image_file = st.file_uploader("UPLOAD IMAGE FILE")

if image_file:
    img = load_image(image_file)
    results = model(img)
    results.save()
    number = len(os.listdir("./runs/detect"))
    if number == 1: number = ""
    st.image(f"./runs/detect/exp{number}/image0.jpg")
    
    
    