from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from typing import List
import base64
import logging
from pathlib import Path
import shutil
from .stitcher import ImageStitcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Image Stitching API",
    description="딥러닝 기반 이미지 스티칭 서비스",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 디렉토리 설정
UPLOAD_DIR = Path("uploads")
RESULT_DIR = Path("results")
UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

# 이미지 스티처 초기화 (평면 파노라마를 위해 cylindrical warping 비활성화)
stitcher = ImageStitcher(use_cylindrical=False)


@app.get("/")
async def root():
    return {
        "message": "AI Image Stitching API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/stitch")
async def stitch_images(images: List[UploadFile] = File(...)):
    """
    여러 이미지를 받아 스티칭하여 파노라마 생성

    Args:
        images: 업로드된 이미지 파일들 (최소 2개)

    Returns:
        JSON: success, result_image (base64), message
    """
    if len(images) < 2:
        raise HTTPException(
            status_code=400,
            detail="최소 2개 이상의 이미지가 필요합니다."
        )

    if len(images) > 10:
        raise HTTPException(
            status_code=400,
            detail="최대 10개까지의 이미지만 처리할 수 있습니다."
        )

    try:
        logger.info(f"{len(images)}개의 이미지 수신")

        # 이미지 읽기 및 검증
        image_arrays = []
        for idx, image_file in enumerate(images):
            # 파일 확장자 체크
            if not image_file.content_type.startswith("image/"):
                raise HTTPException(
                    status_code=400,
                    detail=f"파일 {image_file.filename}은 이미지가 아닙니다."
                )

            # 이미지 읽기
            contents = await image_file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"이미지 {image_file.filename}을 읽을 수 없습니다."
                )

            logger.info(f"이미지 {idx + 1}: {image_file.filename} - 크기: {img.shape}")
            image_arrays.append(img)

        # 이미지 스티칭
        logger.info("이미지 스티칭 시작...")
        result = stitcher.stitch_images(image_arrays)

        if result is None:
            return JSONResponse(
                status_code=200,
                content={
                    "success": False,
                    "message": "이미지 스티칭에 실패했습니다. 이미지들이 겹치는 부분이 충분한지 확인해주세요.",
                    "result_image": None
                }
            )

        # 결과 이미지를 Base64로 인코딩
        _, buffer = cv2.imencode('.png', result)
        result_base64 = base64.b64encode(buffer).decode('utf-8')

        logger.info(f"스티칭 완료! 결과 이미지 크기: {result.shape}")

        return {
            "success": True,
            "result_image": result_base64,
            "message": "이미지 스티칭이 성공적으로 완료되었습니다.",
            "result_shape": {
                "height": int(result.shape[0]),
                "width": int(result.shape[1]),
                "channels": int(result.shape[2]) if len(result.shape) > 2 else 1
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"스티칭 중 오류 발생: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"서버 오류: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
