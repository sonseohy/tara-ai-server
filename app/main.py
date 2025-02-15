from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
from torchvision import models, transforms
from PIL import Image
import os

app = FastAPI()

# 모델 로딩 함수
def load_model():
    model = models.resnet50(pretrained=False)  # 예시로 ResNet50 사용
    model.load_state_dict(torch.load("animal_classifier.pth", map_location="cpu"))  # 모델 파일 경로
    model.eval()  # 모델을 평가 모드로 설정
    return model

# 예측 함수
def predict_animal(model, img_path):
    # 이미지 변환
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # 배치 차원 추가

    # 모델 예측
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    # 예측된 클래스 인덱스가 0, 1, 2, 3에 해당하는 동물 이름
    class_names = ['cong', 'ggam', 'ggang', 'jjong']  # 학습한 클래스들
    return class_names[predicted.item()]  # 클래스 이름 반환

# 모델 로딩
model = load_model()

@app.post("/predict/")
async def predict(files: list[UploadFile] = File(...)):
    results = []
    
    for file in files:
        # 파일 저장 및 예측
        img_path = f"temp/{file.filename}"
        os.makedirs(os.path.dirname(img_path), exist_ok=True)  # temp 폴더 생성 (없으면)
        with open(img_path, "wb") as f:
            f.write(await file.read())
        
        # 동물 예측
        animal_name = predict_animal(model, img_path)
        
        # 결과 저장
        results.append({
            "file_name": file.filename,
            "animal": animal_name if animal_name != "unknown" else "미확인"
        })
        
        # 임시 파일 삭제
        os.remove(img_path)
    
    return JSONResponse(content={"results": results})
