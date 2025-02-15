from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from app.model import load_model, predict_animal
import os

app = FastAPI()

# 모델 로딩
model = load_model()

@app.post("/predict/")
async def predict(files: list[UploadFile] = File(...)):
    results = []
    
    for file in files:
        # 파일 저장 및 예측
        img_path = f"temp/{file.filename}"
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

