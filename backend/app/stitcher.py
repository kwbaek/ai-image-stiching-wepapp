import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Set
import logging
from collections import defaultdict
from .lightglue_matcher import LightGlueMatcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageStitcher:
    """
    Transformer 기반 딥러닝 이미지 스티칭 클래스
    LoFTR (Local Feature Transformer)를 사용하여 이미지를 파노라마로 결합
    """

    def __init__(self, use_affine=True):
        # Transformer 기반 매처 초기화
        self.matcher = LightGlueMatcher()
        self.use_affine = use_affine
        logger.info(f"ImageStitcher initialized with Transformer-based matcher (affine={use_affine}, planar_mode={use_affine})")

    def _cylindrical_warp(self, img: np.ndarray, focal_length: Optional[float] = None) -> np.ndarray:
        """
        이미지를 원통형(cylindrical) 표면에 투영하여 평면 파노라마 생성 준비

        Args:
            img: 입력 이미지
            focal_length: 카메라 초점 거리 (None이면 이미지 너비로 자동 계산)

        Returns:
            원통형 투영된 이미지
        """
        h, w = img.shape[:2]

        # focal length 자동 계산 (이미지 너비 사용)
        if focal_length is None:
            focal_length = w

        # 카메라 내부 파라미터 행렬
        K = np.array([
            [focal_length, 0, w / 2],
            [0, focal_length, h / 2],
            [0, 0, 1]
        ])

        # 역행렬
        K_inv = np.linalg.inv(K)

        # 출력 이미지 좌표 생성
        y_i, x_i = np.indices((h, w))

        # 원통형 좌표로 변환
        # 1. 이미지 좌표를 정규화된 좌표로 변환
        X = np.stack([x_i.ravel(), y_i.ravel(), np.ones_like(x_i.ravel())], axis=0)
        X_norm = K_inv @ X

        x_norm = X_norm[0]
        y_norm = X_norm[1]

        # 2. 원통형 좌표 계산
        theta = np.arctan2(x_norm, 1)  # 수평 각도
        h_cyl = y_norm / np.sqrt(x_norm ** 2 + 1)  # 수직 위치

        # 3. 원통형 좌표를 이미지 좌표로 재변환
        x_cyl = focal_length * theta + w / 2
        y_cyl = focal_length * h_cyl + h / 2

        # 4. 리매핑으로 원통형 투영 적용
        map_x = x_cyl.reshape(h, w).astype(np.float32)
        map_y = y_cyl.reshape(h, w).astype(np.float32)

        warped = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        return warped

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
            # 이미지 전처리: 크기 조정
            processed_images = []
            for i, img in enumerate(images):
                # 너무 큰 이미지는 처리 속도를 위해 리사이즈
                height, width = img.shape[:2]
                max_dimension = 2000  # 더 큰 크기로 유지하여 품질 개선

                if max(height, width) > max_dimension:
                    scale = max_dimension / max(height, width)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
                    logger.info(f"이미지 {i} 리사이즈: {width}x{height} -> {new_width}x{new_height}")

                processed_images.append(img)

            logger.info(f"{len(processed_images)}개의 이미지를 스티칭 중... (planar_mode={'ON' if self.use_affine else 'OFF'})")

            # 자동 배치 인식을 사용한 스티칭
            logger.info("Using auto-layout detection for multi-directional stitching...")
            return self._stitch_auto_layout(processed_images)

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
        두 이미지를 매칭으로 스티칭
        """
        try:
            logger.info("Starting image pair stitching...")
            logger.info(f"Image 1 shape: {img1.shape}, Image 2 shape: {img2.shape}")

            # 특징점 매칭
            src_pts, dst_pts, num_matches = self.matcher.match_images(img1, img2)

            if src_pts is None or dst_pts is None or num_matches < 4:
                logger.error(f"충분한 매칭 포인트를 찾을 수 없습니다. (Found: {num_matches})")
                return None

            logger.info(f"Found {num_matches} matching points")

            # 변환 행렬 계산 (Affine 또는 Homography)
            if self.use_affine:
                logger.info("Computing affine transformation matrix (planar mode)...")
                H = self.matcher.estimate_affine(src_pts, dst_pts)
                transform_type = "Affine"
            else:
                logger.info("Computing homography matrix...")
                H = self.matcher.estimate_homography(src_pts, dst_pts)
                transform_type = "Homography"

            if H is None:
                logger.error(f"{transform_type} 행렬을 계산할 수 없습니다.")
                return None

            logger.info(f"{transform_type} computed successfully")

            # 결과 이미지 크기 계산
            logger.info("Computing output canvas size...")
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

            # 이미지 워핑 및 블렌딩
            output_size = (x_max - x_min, y_max - y_min)
            logger.info(f"Output canvas size: {output_size}")

            # img2를 변환
            logger.info("Warping second image...")
            warped_img2 = cv2.warpPerspective(img2, translation @ H, output_size)

            # img1을 캔버스에 배치
            logger.info("Placing first image on canvas...")
            result = np.zeros((y_max - y_min, x_max - x_min, 3), dtype=np.uint8)
            result[-y_min:-y_min + h1, -x_min:-x_min + w1] = img1

            # 겹치는 영역 찾기 및 블렌딩
            logger.info("Blending overlapping regions...")
            # warped_img2가 0이 아닌 영역을 찾기
            mask2 = (warped_img2.sum(axis=2) > 0).astype(np.uint8)
            mask1 = (result.sum(axis=2) > 0).astype(np.uint8)
            overlap = mask1 & mask2

            # 겹치지 않는 부분은 그대로 복사
            result[mask2 == 1] = warped_img2[mask2 == 1]

            # 겹치는 부분은 평균으로 블렌딩
            overlap_bool = overlap == 1
            if overlap_bool.any():
                result[overlap_bool] = (result[overlap_bool].astype(np.float32) * 0.5 +
                                       warped_img2[overlap_bool].astype(np.float32) * 0.5).astype(np.uint8)

            logger.info("Image stitching completed successfully!")
            return result

        except Exception as e:
            logger.error(f"페어 스티칭 중 오류: {str(e)}")
            return None

    def _match_all_pairs(self, images: List[np.ndarray]) -> Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray, np.ndarray, int]]:
        """
        모든 이미지 쌍에 대해 매칭을 수행하고 변환 행렬 계산

        Args:
            images: 이미지 리스트

        Returns:
            딕셔너리 {(i, j): (src_pts, dst_pts, H, num_matches)}
            H는 affine 또는 homography 변환 행렬 (3x3)
            i < j인 경우만 저장 (중복 방지)
        """
        n = len(images)
        matches = {}

        logger.info(f"Matching all pairs among {n} images...")

        for i in range(n):
            for j in range(i + 1, n):
                try:
                    logger.info(f"Matching image {i} with image {j}...")

                    # 특징점 매칭
                    src_pts, dst_pts, num_matches = self.matcher.match_images(images[i], images[j])

                    if src_pts is not None and dst_pts is not None and num_matches >= 4:
                        # 변환 행렬 계산 (Affine 또는 Homography)
                        if self.use_affine:
                            H = self.matcher.estimate_affine(src_pts, dst_pts)
                            transform_type = "affine"
                        else:
                            H = self.matcher.estimate_homography(src_pts, dst_pts)
                            transform_type = "homography"

                        if H is not None:
                            matches[(i, j)] = (src_pts, dst_pts, H, num_matches)
                            logger.info(f"Successfully matched {i}-{j}: {num_matches} points ({transform_type})")
                        else:
                            logger.warning(f"Failed to compute {transform_type} for {i}-{j}")
                    else:
                        logger.warning(f"Insufficient matches for {i}-{j}: {num_matches}")

                except Exception as e:
                    logger.error(f"Error matching {i}-{j}: {str(e)}")
                    continue

        logger.info(f"Found {len(matches)} valid image pairs")
        return matches

    def _build_match_graph(self, n_images: int, matches: Dict[Tuple[int, int], Tuple]) -> Dict[int, Set[int]]:
        """
        매칭 결과로부터 그래프 구축

        Args:
            n_images: 이미지 개수
            matches: _match_all_pairs의 결과

        Returns:
            인접 리스트 형태의 그래프 {image_id: set of connected image_ids}
        """
        graph = defaultdict(set)

        for (i, j) in matches.keys():
            graph[i].add(j)
            graph[j].add(i)

        # 연결되지 않은 이미지도 초기화
        for i in range(n_images):
            if i not in graph:
                graph[i] = set()

        logger.info(f"Built graph with {len(graph)} nodes")
        for node, neighbors in graph.items():
            logger.info(f"  Image {node}: connected to {len(neighbors)} images {neighbors}")

        return dict(graph)

    def _find_central_image(self, graph: Dict[int, Set[int]]) -> int:
        """
        가장 많은 이미지와 연결된 중심 이미지 찾기

        Args:
            graph: 이미지 연결 그래프

        Returns:
            중심 이미지의 인덱스
        """
        max_connections = -1
        central_idx = 0

        for idx, neighbors in graph.items():
            if len(neighbors) > max_connections:
                max_connections = len(neighbors)
                central_idx = idx

        logger.info(f"Central image: {central_idx} with {max_connections} connections")
        return central_idx

    def _calculate_layout(self, images: List[np.ndarray], matches: Dict[Tuple[int, int], Tuple],
                         graph: Dict[int, Set[int]], central_idx: int) -> Dict[int, np.ndarray]:
        """
        중심 이미지를 기준으로 모든 이미지의 상대적 위치 계산

        Args:
            images: 이미지 리스트
            matches: 매칭 정보
            graph: 연결 그래프
            central_idx: 중심 이미지 인덱스

        Returns:
            {image_idx: cumulative_transformation_matrix}
            변환 행렬은 affine 또는 homography (3x3)
            중심 이미지는 Identity matrix
        """
        n = len(images)
        layout = {}

        # 중심 이미지는 Identity matrix
        layout[central_idx] = np.eye(3)

        # BFS로 중심에서부터 거리 계산
        visited = {central_idx}
        queue = [(central_idx, np.eye(3))]  # (image_idx, cumulative_H)

        while queue:
            current_idx, current_H = queue.pop(0)

            # 인접한 이미지들 탐색
            for neighbor_idx in graph[current_idx]:
                if neighbor_idx in visited:
                    continue

                visited.add(neighbor_idx)

                # current -> neighbor로 가는 homography 찾기
                if (current_idx, neighbor_idx) in matches:
                    _, _, H, _ = matches[(current_idx, neighbor_idx)]
                    # H는 current -> neighbor
                    neighbor_H = H @ current_H
                elif (neighbor_idx, current_idx) in matches:
                    _, _, H, _ = matches[(neighbor_idx, current_idx)]
                    # H는 neighbor -> current이므로 역행렬 필요
                    try:
                        H_inv = np.linalg.inv(H)
                        neighbor_H = H_inv @ current_H
                    except np.linalg.LinAlgError:
                        logger.error(f"Failed to invert homography for {neighbor_idx}-{current_idx}")
                        continue
                else:
                    logger.error(f"No match found between {current_idx} and {neighbor_idx}")
                    continue

                layout[neighbor_idx] = neighbor_H
                queue.append((neighbor_idx, neighbor_H))
                logger.info(f"Added image {neighbor_idx} to layout via {current_idx}")

        logger.info(f"Layout calculated for {len(layout)} images")
        return layout

    def _stitch_auto_layout(self, images: List[np.ndarray]) -> Optional[np.ndarray]:
        """
        자동 배치 인식을 사용한 스티칭
        모든 이미지 쌍을 매칭하여 최적의 배치 찾기

        Args:
            images: 스티칭할 이미지 리스트

        Returns:
            스티칭된 파노라마 이미지
        """
        if len(images) < 2:
            return None

        # 이미지가 2개인 경우 기존 방식 사용
        if len(images) == 2:
            logger.info("Only 2 images, using simple pair stitching")
            return self._stitch_pair(images[0], images[1])

        try:
            # 1. 모든 이미지 쌍 매칭
            matches = self._match_all_pairs(images)

            if len(matches) == 0:
                logger.error("No valid matches found between any image pairs")
                return None

            # 2. 매칭 그래프 구축
            graph = self._build_match_graph(len(images), matches)

            # 3. 중심 이미지 찾기
            central_idx = self._find_central_image(graph)

            # 4. 각 이미지의 상대적 위치 계산
            layout = self._calculate_layout(images, matches, graph, central_idx)

            if len(layout) < len(images):
                logger.warning(f"Only {len(layout)}/{len(images)} images could be positioned. Some images may be disconnected.")

            # 5. 모든 이미지를 하나의 캔버스에 배치
            return self._stitch_with_layout(images, layout)

        except Exception as e:
            logger.error(f"Auto-layout stitching failed: {str(e)}")
            logger.info("Falling back to sequential stitching...")
            return self._stitch_basic(images)

    def _stitch_with_layout(self, images: List[np.ndarray], layout: Dict[int, np.ndarray]) -> Optional[np.ndarray]:
        """
        계산된 배치 정보를 사용하여 모든 이미지를 하나의 캔버스에 스티칭

        Args:
            images: 이미지 리스트
            layout: 각 이미지의 homography matrix

        Returns:
            스티칭된 결과 이미지
        """
        try:
            # 1. 전체 캔버스 크기 계산
            logger.info("Calculating canvas size...")
            all_corners = []

            for idx, H in layout.items():
                h, w = images[idx].shape[:2]
                corners = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
                transformed_corners = cv2.perspectiveTransform(corners, H)
                all_corners.append(transformed_corners)

            all_corners = np.concatenate(all_corners, axis=0)
            [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
            [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

            # Translation matrix to shift everything to positive coordinates
            translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])

            output_width = x_max - x_min
            output_height = y_max - y_min

            logger.info(f"Canvas size: {output_width}x{output_height}")

            # 2. 빈 캔버스 생성
            canvas = np.zeros((output_height, output_width, 3), dtype=np.uint8)
            weight_map = np.zeros((output_height, output_width), dtype=np.float32)

            # 3. 모든 이미지를 캔버스에 배치 및 블렌딩
            logger.info("Placing all images on canvas...")
            for idx, H in layout.items():
                logger.info(f"Placing image {idx}...")

                # Apply translation to homography
                final_H = translation @ H

                # Warp image
                warped = cv2.warpPerspective(images[idx], final_H, (output_width, output_height))

                # Create mask for non-zero pixels
                mask = (warped.sum(axis=2) > 0).astype(np.float32)

                # Accumulate weighted sum
                for c in range(3):
                    canvas[:, :, c] = canvas[:, :, c].astype(np.float32) + warped[:, :, c].astype(np.float32) * mask

                weight_map += mask

            # 4. Normalize by weight
            logger.info("Blending images...")
            for c in range(3):
                canvas[:, :, c] = np.divide(
                    canvas[:, :, c],
                    weight_map,
                    out=np.zeros_like(canvas[:, :, c], dtype=np.float32),
                    where=weight_map > 0
                ).astype(np.uint8)

            logger.info("Auto-layout stitching completed successfully!")
            return canvas

        except Exception as e:
            logger.error(f"Error in _stitch_with_layout: {str(e)}")
            return None
