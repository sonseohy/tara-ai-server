# Python 3.10 이미지를 기반으로 설정
FROM python:3.10-slim

# 작업 디렉토리 생성
WORKDIR /app

# 필수 파일들 복사
COPY requirements.txt .

# 필요 라이브러리 설치
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# 모델 파일도 포함되도록 하기 위해 COPY 추가
COPY animal_classifier.pth /app/animal_classifier.pth

# FastAPI 서버 실행 (main.py로 변경)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5050"]
