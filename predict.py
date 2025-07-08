from PIL import Image
import torch
from torchvision import transforms

# 推論用の画像前処理（val/testと同じ）
predict_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

def predict_image(img_path_or_file, model, classes, device):
    image = Image.open(img_path_or_file).convert('RGB')
    input_tensor = predict_transform(image).unsqueeze(0).to(device)  # shape: [1, 3, 224, 224]
    
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
    
    return classes[predicted.item()]
