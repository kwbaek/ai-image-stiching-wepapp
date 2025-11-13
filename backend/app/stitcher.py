import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageStitcher:
    """
    OpenCV 기반 이미지 스티칭 클래스
    SIFT 특징점 검출 및 매칭을 사용하여 이미지를 파노라마로 결합
    """

    def __init__(self):
        self.detector = cv2.SIFT_create()

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
        두 이미지를 스티칭
        """
        try:
            # SIFT 특징점 검출
            kp1, des1 = self.detector.detectAndCompute(img1, None)
            kp2, des2 = self.detector.detectAndCompute(img2, None)

            if des1 is None or des2 is None:
                logger.error("특징점을 찾을 수 없습니다.")
                return None

            # BFMatcher로 특징점 매칭
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)

            # Lowe's ratio test로 좋은 매칭만 선택
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

            if len(good_matches) < 10:
                logger.error("충분한 매칭 포인트를 찾을 수 없습니다.")
                return None

            # 호모그래피 계산
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

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
