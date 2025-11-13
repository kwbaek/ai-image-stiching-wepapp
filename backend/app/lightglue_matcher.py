"""
LightGlue Transformer 기반 Feature Matcher
SuperPoint + LightGlue를 사용한 딥러닝 기반 이미지 매칭
"""

import torch
import cv2
import numpy as np
from typing import Tuple, Optional
import logging

try:
    from kornia.feature import LoFTR
    KORNIA_AVAILABLE = True
except ImportError:
    KORNIA_AVAILABLE = False
    logging.warning("Kornia not available. Using fallback feature matcher.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LightGlueMatcher:
    """
    Transformer 기반 feature matching을 위한 클래스
    LoFTR (Local Feature TRansformer) 사용
    """

    def __init__(self, device: str = None):
        """
        Args:
            device: 'cuda' 또는 'cpu'. None이면 자동 선택
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")

        # LoFTR 모델 로드 (Transformer 기반)
        if KORNIA_AVAILABLE:
            try:
                self.matcher = LoFTR(pretrained='outdoor')
                self.matcher = self.matcher.to(self.device)
                self.matcher.eval()
                self.use_deep_learning = True
                logger.info("LoFTR Transformer model loaded successfully!")
            except Exception as e:
                logger.warning(f"Failed to load LoFTR: {e}. Using SIFT fallback.")
                self.use_deep_learning = False
                self.sift = cv2.SIFT_create()
        else:
            self.use_deep_learning = False
            self.sift = cv2.SIFT_create()
            logger.info("Using SIFT as fallback feature detector")

    def match_images(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        confidence_threshold: float = 0.2
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
        """
        두 이미지 간의 매칭 포인트 찾기

        Args:
            img1: 첫 번째 이미지 (numpy array)
            img2: 두 번째 이미지 (numpy array)
            confidence_threshold: 매칭 신뢰도 임계값 (0-1)

        Returns:
            src_pts: 첫 번째 이미지의 매칭 포인트
            dst_pts: 두 번째 이미지의 매칭 포인트
            num_matches: 매칭 포인트 개수
        """
        if self.use_deep_learning:
            return self._match_with_loftr(img1, img2, confidence_threshold)
        else:
            return self._match_with_sift(img1, img2)

    def _match_with_loftr(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        confidence_threshold: float
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
        """
        LoFTR Transformer 모델을 사용한 매칭
        """
        try:
            # 이미지를 grayscale로 변환
            if len(img1.shape) == 3:
                gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            else:
                gray1 = img1

            if len(img2.shape) == 3:
                gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            else:
                gray2 = img2

            # 이미지를 텐서로 변환 [1, 1, H, W]
            tensor1 = torch.from_numpy(gray1).float()[None, None] / 255.0
            tensor2 = torch.from_numpy(gray2).float()[None, None] / 255.0

            tensor1 = tensor1.to(self.device)
            tensor2 = tensor2.to(self.device)

            # LoFTR로 매칭 수행
            with torch.no_grad():
                input_dict = {
                    "image0": tensor1,
                    "image1": tensor2
                }
                correspondences = self.matcher(input_dict)

            # 매칭 포인트 추출
            mkpts0 = correspondences['keypoints0'].cpu().numpy()
            mkpts1 = correspondences['keypoints1'].cpu().numpy()
            confidence = correspondences['confidence'].cpu().numpy()

            # 신뢰도 필터링
            mask = confidence > confidence_threshold
            mkpts0 = mkpts0[mask]
            mkpts1 = mkpts1[mask]

            num_matches = len(mkpts0)
            logger.info(f"LoFTR found {num_matches} confident matches")

            if num_matches < 4:
                logger.warning("Not enough matches found with LoFTR")
                return None, None, 0

            # OpenCV 형식으로 변환 [N, 1, 2]
            src_pts = mkpts0.reshape(-1, 1, 2).astype(np.float32)
            dst_pts = mkpts1.reshape(-1, 1, 2).astype(np.float32)

            return src_pts, dst_pts, num_matches

        except Exception as e:
            logger.error(f"LoFTR matching failed: {e}")
            # Fallback to SIFT
            return self._match_with_sift(img1, img2)

    def _match_with_sift(
        self,
        img1: np.ndarray,
        img2: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
        """
        SIFT를 사용한 전통적인 매칭 (Fallback)
        """
        try:
            # Grayscale 변환
            if len(img1.shape) == 3:
                gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            else:
                gray1 = img1

            if len(img2.shape) == 3:
                gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            else:
                gray2 = img2

            # SIFT 특징점 검출
            kp1, des1 = self.sift.detectAndCompute(gray1, None)
            kp2, des2 = self.sift.detectAndCompute(gray2, None)

            if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
                logger.error("Not enough keypoints detected")
                return None, None, 0

            # BFMatcher로 매칭
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)

            # Lowe's ratio test
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

            num_matches = len(good_matches)
            logger.info(f"SIFT found {num_matches} good matches")

            if num_matches < 4:
                logger.warning("Not enough good matches found")
                return None, None, 0

            # 매칭 포인트 추출
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            return src_pts, dst_pts, num_matches

        except Exception as e:
            logger.error(f"SIFT matching failed: {e}")
            return None, None, 0

    def estimate_homography(
        self,
        src_pts: np.ndarray,
        dst_pts: np.ndarray,
        ransac_threshold: float = 5.0
    ) -> Optional[np.ndarray]:
        """
        RANSAC를 사용하여 호모그래피 행렬 추정

        Args:
            src_pts: 소스 이미지 포인트
            dst_pts: 목적지 이미지 포인트
            ransac_threshold: RANSAC 임계값

        Returns:
            호모그래피 행렬 (3x3) 또는 None
        """
        try:
            H, mask = cv2.findHomography(
                dst_pts,
                src_pts,
                cv2.RANSAC,
                ransac_threshold
            )

            if H is None:
                logger.error("Failed to compute homography")
                return None

            # Inlier 비율 체크
            inlier_ratio = np.sum(mask) / len(mask)
            logger.info(f"Homography inlier ratio: {inlier_ratio:.2%}")

            if inlier_ratio < 0.1:
                logger.warning("Low inlier ratio, homography may be unreliable")

            return H

        except Exception as e:
            logger.error(f"Homography estimation failed: {e}")
            return None
