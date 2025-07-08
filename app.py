# app.py
import streamlit as st
from PIL import Image
import torch
import torchvision.models as models
import torch.nn as nn
from predict import predict_image
from pokemon_zukan import classes

# モデルの構築（学習時と同じ構造）
net = models.vgg16_bn(pretrained=False)
in_features = net.classifier[6].in_features
net.classifier[6] = nn.Linear(in_features, len(classes))
net.load_state_dict(torch.load('model.pth', map_location='cpu'))
net.eval()

# Streamlit UI
st.title("ポケモン画像分類アプリ")
uploaded_file = st.file_uploader("画像を選択してください", type=['jpg', 'png'])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='アップロードされた画像', use_column_width=True)

    result = predict_image(uploaded_file, net, classes, device='cpu')
    st.success(f'このポケモンは「{result}」です！')
