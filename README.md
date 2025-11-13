# AI Image Stitching Web App

딥러닝 기반 이미지 스티칭 웹 애플리케이션입니다. SuperGlue Transformer 모델을 활용하여 여러 이미지를 자동으로 파노라마로 합성합니다.

## 기술 스택

### Frontend
- React 18 + TypeScript
- Vite
- TailwindCSS
- Axios

### Backend
- Python 3.9+
- FastAPI
- PyTorch
- OpenCV
- SuperGlue (Transformer 기반 feature matching)

## 주요 기능

- 🖼️ 다중 이미지 업로드
- 🤖 AI 기반 자동 이미지 매칭 및 정렬
- 🔄 실시간 스티칭 진행 상황 표시
- 📥 결과 이미지 다운로드
- 🎨 인터랙티브 UI

## 설치 및 실행

### Backend 설정

```bash
cd backend
python -m venv venv
source venv/bin/activate  # macOS
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Frontend 설정

```bash
cd frontend
npm install
npm run dev
```

## 프로젝트 구조

```
ai-image-stitching-webapp/
├── frontend/          # React 프론트엔드
│   ├── src/
│   │   ├── components/
│   │   ├── services/
│   │   └── App.tsx
│   └── package.json
├── backend/           # FastAPI 백엔드
│   ├── app/
│   │   ├── main.py
│   │   ├── models/
│   │   └── services/
│   └── requirements.txt
└── README.md
```

## 사용 방법

1. 웹 브라우저에서 `http://localhost:5173` 접속
2. 여러 이미지를 드래그 앤 드롭 또는 선택하여 업로드
3. "Stitch Images" 버튼 클릭
4. AI가 자동으로 이미지를 분석하고 파노라마 생성
5. 결과 이미지 다운로드

## 라이선스

MIT
