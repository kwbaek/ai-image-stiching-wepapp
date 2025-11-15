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
            device: 'cuda', 'mps', 또는 'cpu'. None이면 자동 선택
        """
        if device is None:
            # Apple Silicon MPS (Metal Performance Shaders) 지원
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")

        # SIFT를 항상 초기화 (fallback용)
        self.sift = cv2.SIFT_create()
        logger.info("SIFT feature detector initialized as fallback")

        # CPU에서는 LoFTR가 너무 느리므로 SIFT만 사용
        if self.device.type == 'cpu':
            logger.info("CPU detected - using SIFT for faster processing")
            self.use_deep_learning = False
            return

        # GPU (CUDA 또는 MPS)에서 LoFTR 모델 로드 (Transformer 기반)
        if KORNIA_AVAILABLE:
            try:
                logger.info("Loading LoFTR Transformer model...")
                self.matcher = LoFTR(pretrained='outdoor')
                self.matcher = self.matcher.to(self.device)
                self.matcher.eval()
                self.use_deep_learning = True
                logger.info(f"LoFTR Transformer model loaded successfully on {self.device}!")
            except Exception as e:
                logger.warning(f"Failed to load LoFTR on {self.device}: {e}. Using SIFT fallback.")
                self.use_deep_learning = False
        else:
            self.use_deep_learning = False
            logger.info("Using SIFT as fallback feature detector")

    def match_images(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        confidence_threshold: float = 0.7
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
        """
        두 이미지 간의 매칭 포인트 찾기

        Args:
            img1: 첫 번째 이미지 (numpy array)
            img2: 두 번째 이미지 (numpy array)
            confidence_threshold: 매칭 신뢰도 임계값 (0-1, 기본값 0.7로 고품질 매칭만 선택)

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
        # 원본 이미지 백업 (SIFT fallback용)
        orig_img1 = img1
        orig_img2 = img2

        try:
            # 메모리 절약을 위해 이미지 크기 조정
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]
            max_dim = 840  # LoFTR에서 권장하는 크기 (840x840)

            scale1 = 1.0
            if max(h1, w1) > max_dim:
                scale1 = max_dim / max(h1, w1)
                new_w1 = int(w1 * scale1)
                new_h1 = int(h1 * scale1)
                img1 = cv2.resize(img1, (new_w1, new_h1), interpolation=cv2.INTER_AREA)
                logger.info(f"Resized image1 for LoFTR: {w1}x{h1} -> {new_w1}x{new_h1}")

            scale2 = 1.0
            if max(h2, w2) > max_dim:
                scale2 = max_dim / max(h2, w2)
                new_w2 = int(w2 * scale2)
                new_h2 = int(h2 * scale2)
                img2 = cv2.resize(img2, (new_w2, new_h2), interpolation=cv2.INTER_AREA)
                logger.info(f"Resized image2 for LoFTR: {w2}x{h2} -> {new_w2}x{new_h2}")

            # 이미지를 grayscale로 변환
            if len(img1.shape) == 3:
                gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            else:
                gray1 = img1

            if len(img2.shape) == 3:
                gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            else:
                gray2 = img2

            # MPS 메모리 정리
            if self.device.type == 'mps':
                torch.mps.empty_cache()
                logger.info("Cleared MPS cache before processing")

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

            # 텐서 메모리 해제
            del tensor1, tensor2, correspondences
            if self.device.type == 'mps':
                torch.mps.empty_cache()
            elif self.device.type == 'cuda':
                torch.cuda.empty_cache()

            # 스케일 조정을 역으로 적용
            if scale1 != 1.0:
                mkpts0 = mkpts0 / scale1
            if scale2 != 1.0:
                mkpts1 = mkpts1 / scale2

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
            # 메모리 정리
            if self.device.type == 'mps':
                torch.mps.empty_cache()
            elif self.device.type == 'cuda':
                torch.cuda.empty_cache()
            # Fallback to SIFT with original images
            logger.info("Falling back to SIFT matcher...")
            return self._match_with_sift(orig_img1, orig_img2)

    def _match_with_sift(
        self,
        img1: np.ndarray,
        img2: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
        """
        SIFT를 사용한 전통적인 매칭 (Fallback)
        """
        try:
            logger.info("Starting SIFT feature detection...")

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
            logger.info("Detecting keypoints in image 1...")
            kp1, des1 = self.sift.detectAndCompute(gray1, None)
            logger.info(f"Found {len(kp1)} keypoints in image 1")

            logger.info("Detecting keypoints in image 2...")
            kp2, des2 = self.sift.detectAndCompute(gray2, None)
            logger.info(f"Found {len(kp2)} keypoints in image 2")

            if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
                logger.error("Not enough keypoints detected")
                return None, None, 0

            # BFMatcher로 매칭
            logger.info("Matching features with BFMatcher...")
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)
            logger.info(f"Found {len(matches)} initial matches")

            # Lowe's ratio test
            logger.info("Applying Lowe's ratio test to filter matches...")
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

    def estimate_affine(
        self,
        src_pts: np.ndarray,
        dst_pts: np.ndarray,
        ransac_threshold: float = 3.0
    ) -> Optional[np.ndarray]:
        """
        RANSAC를 사용하여 어파인 변환 행렬 추정 (평면 스티칭용)

        Args:
            src_pts: 소스 이미지 포인트
            dst_pts: 목적지 이미지 포인트
            ransac_threshold: RANSAC 임계값 (픽셀 단위, 기본값 3.0)

        Returns:
            어파인 행렬 (2x3) 또는 None
        """
        try:
            # Affine transformation 추정 (6 DOF: 회전, 이동, 스케일, 전단)
            # 원근 왜곡 없이 평면으로 스티칭
            M, mask = cv2.estimateAffinePartial2D(
                dst_pts,
                src_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=ransac_threshold
            )

            if M is None:
                logger.warning("Affine estimation failed, trying full affine...")
                # Partial2D가 실패하면 full affine 시도
                M, mask = cv2.estimateAffine2D(
                    dst_pts,
                    src_pts,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=ransac_threshold
                )

            if M is None:
                logger.error("Failed to compute affine transformation")
                return None

            # Inlier 비율 체크
            inlier_ratio = np.sum(mask) / len(mask)
            logger.info(f"Affine inlier ratio: {inlier_ratio:.2%}")

            if inlier_ratio < 0.3:
                logger.warning(f"Low inlier ratio ({inlier_ratio:.2%}), result may be unreliable.")

            # 2x3 행렬을 3x3 homogeneous 형태로 변환 (호환성 위해)
            M_3x3 = np.vstack([M, [0, 0, 1]])
            return M_3x3

        except Exception as e:
            logger.error(f"Affine estimation failed: {e}")
            return None

    def estimate_homography(
        self,
        src_pts: np.ndarray,
        dst_pts: np.ndarray,
        ransac_threshold: float = 3.0
    ) -> Optional[np.ndarray]:
        """
        RANSAC를 사용하여 호모그래피 행렬 추정

        Args:
            src_pts: 소스 이미지 포인트
            dst_pts: 목적지 이미지 포인트
            ransac_threshold: RANSAC 임계값 (픽셀 단위, 기본값 3.0)

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

            if inlier_ratio < 0.3:
                logger.warning(f"Low inlier ratio ({inlier_ratio:.2%}), result may be unreliable. Consider using images with more overlap.")

            return H

        except Exception as e:
            logger.error(f"Homography estimation failed: {e}")
            return None
