# AI Image Stitching Web App

딥러닝 기반 이미지 스티칭 웹 애플리케이션입니다. **LoFTR (Local Feature Transformer)** 모델을 활용하여 여러 이미지를 자동으로 파노라마로 합성합니다.

## 🤖 AI 모델

- **LoFTR (Local Feature TRansformer)**: Transformer 기반의 최신 feature matching 모델
- 전통적인 SIFT보다 **더 정확하고 강건한** 이미지 매칭
- 딥러닝 기반으로 복잡한 변환과 조명 변화에도 우수한 성능
- GPU 가속 지원 (CUDA 사용 가능 시 자동으로 GPU 사용)

## 기술 스택

### Frontend
- React 18 + TypeScript
- Vite
- TailwindCSS
- Axios

### Backend
- Python 3.9+
- FastAPI
- PyTorch (딥러닝 프레임워크)
- Kornia (Computer Vision 라이브러리)
- OpenCV
- LoFTR (Transformer 기반 feature matching)

## 주요 기능

- 🖼️ 다중 이미지 업로드
- 🤖 AI 기반 자동 이미지 매칭 및 정렬
- 🔄 실시간 스티칭 진행 상황 표시
- 📥 결과 이미지 다운로드
- 🎨 인터랙티브 UI

## 설치 및 실행

### 빠른 시작 (macOS)

**1. 자동 설치**
```bash
./setup.sh
```

**2. 백엔드 실행 (터미널 1)**
```bash
./run-backend.sh
# 또는 수동으로:
# cd backend
# source venv/bin/activate
# uvicorn app.main:app --reload --port 8000
```

**3. 프론트엔드 실행 (터미널 2)**
```bash
./run-frontend.sh
# 또는 수동으로:
# cd frontend
# npm run dev
```

**4. 브라우저에서 열기**
```
http://localhost:5173
```

### 수동 설치

#### Backend 설정

```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

#### Frontend 설정

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
3. "이미지 합성하기" 버튼 클릭
4. **AI Transformer 모델**이 자동으로 이미지를 분석하고 파노라마 생성
   - 첫 실행 시 LoFTR 모델을 다운로드합니다 (인터넷 연결 필요)
   - GPU가 있으면 자동으로 CUDA 가속을 사용합니다
5. 결과 이미지 다운로드

### 팁
- 겹치는 부분이 30% 이상인 이미지를 사용하세요
- **이미지 순서는 자유롭게!** AI가 자동으로 배치를 파악합니다
  - 좌우, 상하, 복잡한 그리드 배치 모두 지원
  - 어떤 순서로 업로드해도 자동으로 올바른 위치를 찾습니다
- 조명이나 각도가 크게 다른 이미지도 AI가 잘 처리합니다
- GPU가 없어도 CPU에서 동작합니다 (다소 느릴 수 있음)

### 자동 배치 인식 (Auto-Layout Detection)
- **모든 방향 지원**: 좌우(가로), 상하(세로), 대각선 등 모든 방향의 이미지 스티칭 가능
- **순서 무관**: 10장의 이미지를 어떤 순서로 올려도 자동으로 배치 파악
- **복잡한 레이아웃**: 2x5, 3x3 등 복잡한 그리드 배치도 자동 처리
- **지능형 매칭**: 모든 이미지 쌍을 분석하여 최적의 연결 관계 파악
- **중심 기반 확장**: 가장 많은 이미지와 연결된 중심 이미지부터 시작하여 안정적으로 확장

## 라이선스

MIT
