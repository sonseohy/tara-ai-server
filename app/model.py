import torch
from torchvision import models, transforms
from PIL import Image

# 모델 로딩
def load_model():
    model = models.resnet50(pretrained=False)  # 모델 정의
    model.load_state_dict(torch.load("animal_classifier.pth"))
    model.eval()
    return model

# 이미지 전처리
def process_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path)
    img = transform(img).unsqueeze(0)
    return img

# 예측 함수
def predict_animal(model, image_path):
    image = process_image(image_path)
    with torch.no_grad():
        outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    
    # 클래스 맵핑 (0, 1, 2, ... 값에 해당하는 동물 이름)
    class_names = ['cong', 'ggam', 'ggang', 'jjong']  # 학습한 동물 클래스 이름
    return class_names[predicted.item()] if predicted.item() < len(class_names) else "unknown"
