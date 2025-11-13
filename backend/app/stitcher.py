import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging
from .lightglue_matcher import LightGlueMatcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageStitcher:
    """
    Transformer 기반 딥러닝 이미지 스티칭 클래스
    LoFTR (Local Feature Transformer)를 사용하여 이미지를 파노라마로 결합
    """

    def __init__(self):
        # Transformer 기반 매처 초기화
        self.matcher = LightGlueMatcher()
        logger.info("ImageStitcher initialized with Transformer-based matcher")

    def stitch_images(self, images: List[np.ndarray]) -> Optional[np.ndarray]:
        """
        여러 이미지를 스티칭하여 파노라마 생성

        Args:
            images: 스티칭할 이미지 리스트 (numpy array)

        Returns:
            스티칭된 파노라마 이미지, 실패 시 None
        """
        if len(images) < 2:
            logger.error("최소 2개 이상의 이미지가 필요합니다.")
            return None

        try:
            # OpenCV Stitcher 사용 (고급 기능)
            stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)

            # 이미지 전처리: 크기 조정
            processed_images = []
            for img in images:
                # 너무 큰 이미지는 처리 속도를 위해 리사이즈
                height, width = img.shape[:2]
                max_dimension = 1000

                if max(height, width) > max_dimension:
                    scale = max_dimension / max(height, width)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    img = cv2.resize(img, (new_width, new_height))

                processed_images.append(img)

            logger.info(f"{len(processed_images)}개의 이미지를 스티칭 중...")

            # 스티칭 수행
            status, panorama = stitcher.stitch(processed_images)

            if status == cv2.Stitcher_OK:
                logger.info("스티칭 성공!")
                return panorama
            else:
                logger.error(f"스티칭 실패. 상태 코드: {status}")
                # Fallback: 기본 방식으로 시도
                return self._stitch_basic(processed_images)

        except Exception as e:
            logger.error(f"스티칭 중 오류 발생: {str(e)}")
            # Fallback: 기본 방식으로 시도
            try:
                return self._stitch_basic(images)
            except Exception as e2:
                logger.error(f"Fallback 스티칭도 실패: {str(e2)}")
                return None

    def _stitch_basic(self, images: List[np.ndarray]) -> Optional[np.ndarray]:
        """
        기본적인 특징점 매칭을 사용한 스티칭 (Fallback)
        """
        if len(images) < 2:
            return None

        result = images[0]

        for i in range(1, len(images)):
            result = self._stitch_pair(result, images[i])
            if result is None:
                logger.error(f"이미지 페어 {i}의 스티칭 실패")
                return None

        return result

    def _stitch_pair(self, img1: np.ndarray, img2: np.ndarray) -> Optional[np.ndarray]:
        """
        두 이미지를 Transformer 기반 매칭으로 스티칭
        """
        try:
            logger.info("Using Transformer-based matching for image pair...")

            # Transformer 기반 매칭
            src_pts, dst_pts, num_matches = self.matcher.match_images(img1, img2)

            if src_pts is None or dst_pts is None or num_matches < 4:
                logger.error(f"충분한 매칭 포인트를 찾을 수 없습니다. (Found: {num_matches})")
                return None

            logger.info(f"Found {num_matches} matching points using Transformer model")

            # 호모그래피 계산
            H = self.matcher.estimate_homography(src_pts, dst_pts)

            if H is None:
                logger.error("호모그래피 행렬을 계산할 수 없습니다.")
                return None

            # 결과 이미지 크기 계산
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]

            corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
            corners2_transformed = cv2.perspectiveTransform(corners2, H)

            corners = np.concatenate((
                np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2),
                corners2_transformed
            ), axis=0)

            [x_min, y_min] = np.int32(corners.min(axis=0).ravel() - 0.5)
            [x_max, y_max] = np.int32(corners.max(axis=0).ravel() + 0.5)

            translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])

            # 이미지 워핑
            result = cv2.warpPerspective(img2, translation @ H, (x_max - x_min, y_max - y_min))
            result[-y_min:-y_min + h1, -x_min:-x_min + w1] = img1

            return result

        except Exception as e:
            logger.error(f"페어 스티칭 중 오류: {str(e)}")
            return None
