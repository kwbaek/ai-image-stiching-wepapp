import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Set
import logging
from collections import defaultdict
from io import BytesIO
import cairosvg
from lxml import etree
import svgwrite
from PIL import Image
from .lightglue_matcher import LightGlueMatcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SVGStitcher:
    """
    SVG 파일 스티칭 클래스
    Transformer 기반 이미지 매칭과 SVG 벡터 변환을 결합하여 정밀한 파노라마 생성
    """

    def __init__(self, raster_dpi: int = 300):
        """
        Args:
            raster_dpi: SVG를 래스터화할 때 사용할 DPI (해상도)
        """
        self.matcher = LightGlueMatcher()
        self.raster_dpi = raster_dpi
        logger.info(f"SVGStitcher initialized with DPI={raster_dpi}")

    def _svg_to_png(self, svg_bytes: bytes, dpi: int = 300) -> np.ndarray:
        """
        SVG를 고해상도 PNG로 래스터화

        Args:
            svg_bytes: SVG 파일 바이트
            dpi: 래스터화 DPI

        Returns:
            OpenCV 이미지 (numpy array)
        """
        try:
            # CairoSVG를 사용하여 PNG로 변환
            png_bytes = cairosvg.svg2png(bytestring=svg_bytes, dpi=dpi)

            # PIL로 읽기
            pil_image = Image.open(BytesIO(png_bytes))

            # OpenCV 포맷으로 변환 (RGB -> BGR)
            img_array = np.array(pil_image)
            if len(img_array.shape) == 2:  # 그레이스케일
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
            elif img_array.shape[2] == 4:  # RGBA
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
            else:  # RGB
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            logger.info(f"SVG rasterized to PNG: {img_array.shape}")
            return img_array

        except Exception as e:
            logger.error(f"SVG to PNG conversion failed: {str(e)}")
            raise

    def _parse_svg(self, svg_bytes: bytes) -> Tuple[etree.Element, float, float]:
        """
        SVG 파일 파싱 및 크기 추출

        Args:
            svg_bytes: SVG 파일 바이트

        Returns:
            (SVG root element, width, height)
        """
        try:
            tree = etree.fromstring(svg_bytes)

            # SVG 크기 추출
            width = tree.get('width')
            height = tree.get('height')

            # viewBox에서 크기 추출 시도
            if width is None or height is None:
                viewbox = tree.get('viewBox')
                if viewbox:
                    parts = viewbox.split()
                    width = float(parts[2])
                    height = float(parts[3])
                else:
                    # 기본값 설정
                    width = 512
                    height = 512
                    logger.warning(f"SVG dimensions not found, using default {width}x{height}")
            else:
                # 단위 제거 (예: "512px" -> 512)
                width = float(str(width).replace('px', '').replace('pt', ''))
                height = float(str(height).replace('px', '').replace('pt', ''))

            logger.info(f"Parsed SVG: {width}x{height}")
            return tree, width, height

        except Exception as e:
            logger.error(f"SVG parsing failed: {str(e)}")
            raise

    def _match_all_pairs(self, images: List[np.ndarray]) -> Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray, np.ndarray, int]]:
        """
        모든 이미지 쌍에 대해 매칭 수행

        Args:
            images: 래스터화된 이미지 리스트

        Returns:
            딕셔너리 {(i, j): (src_pts, dst_pts, H, num_matches)}
        """
        n = len(images)
        matches = {}

        logger.info(f"Matching all pairs among {n} SVG images...")

        for i in range(n):
            for j in range(i + 1, n):
                try:
                    logger.info(f"Matching SVG {i} with SVG {j}...")

                    # 특징점 매칭
                    src_pts, dst_pts, num_matches = self.matcher.match_images(images[i], images[j])

                    if src_pts is not None and dst_pts is not None and num_matches >= 4:
                        # Homography 계산
                        H = self.matcher.estimate_homography(src_pts, dst_pts)

                        if H is not None:
                            matches[(i, j)] = (src_pts, dst_pts, H, num_matches)
                            logger.info(f"Successfully matched {i}-{j}: {num_matches} points")
                        else:
                            logger.warning(f"Failed to compute homography for {i}-{j}")
                    else:
                        logger.warning(f"Insufficient matches for {i}-{j}: {num_matches}")

                except Exception as e:
                    logger.error(f"Error matching {i}-{j}: {str(e)}")
                    continue

        logger.info(f"Found {len(matches)} valid SVG pairs")
        return matches

    def _build_match_graph(self, n_images: int, matches: Dict[Tuple[int, int], Tuple]) -> Dict[int, Set[int]]:
        """
        매칭 결과로부터 그래프 구축
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
        return dict(graph)

    def _find_central_image(self, graph: Dict[int, Set[int]]) -> int:
        """
        가장 많은 이미지와 연결된 중심 이미지 찾기
        """
        max_connections = -1
        central_idx = 0

        for idx, neighbors in graph.items():
            if len(neighbors) > max_connections:
                max_connections = len(neighbors)
                central_idx = idx

        logger.info(f"Central SVG: {central_idx} with {max_connections} connections")
        return central_idx

    def _calculate_layout(self, images: List[np.ndarray], matches: Dict[Tuple[int, int], Tuple],
                         graph: Dict[int, Set[int]], central_idx: int) -> Dict[int, np.ndarray]:
        """
        중심 이미지를 기준으로 모든 이미지의 상대적 위치(homography) 계산
        """
        layout = {}
        layout[central_idx] = np.eye(3)

        visited = {central_idx}
        queue = [(central_idx, np.eye(3))]

        while queue:
            current_idx, current_H = queue.pop(0)

            for neighbor_idx in graph[current_idx]:
                if neighbor_idx in visited:
                    continue

                visited.add(neighbor_idx)

                if (current_idx, neighbor_idx) in matches:
                    _, _, H, _ = matches[(current_idx, neighbor_idx)]
                    neighbor_H = H @ current_H
                elif (neighbor_idx, current_idx) in matches:
                    _, _, H, _ = matches[(neighbor_idx, current_idx)]
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
                logger.info(f"Added SVG {neighbor_idx} to layout via {current_idx}")

        logger.info(f"Layout calculated for {len(layout)} SVG images")
        return layout

    def _homography_to_svg_transform(self, H: np.ndarray) -> str:
        """
        Homography 행렬을 SVG transform 문자열로 변환

        Args:
            H: 3x3 Homography 행렬

        Returns:
            SVG transform 문자열 (예: "matrix(a,b,c,d,e,f)")
        """
        # Homography를 Affine 근사 (SVG는 affine transform만 지원)
        # H = [[h00, h01, h02],
        #      [h10, h11, h12],
        #      [h20, h21, h22]]

        # SVG matrix(a, b, c, d, e, f)
        # a=h00, b=h10, c=h01, d=h11, e=h02, f=h12
        a = H[0, 0]
        b = H[1, 0]
        c = H[0, 1]
        d = H[1, 1]
        e = H[0, 2]
        f = H[1, 2]

        return f"matrix({a},{b},{c},{d},{e},{f})"

    def _create_stitched_svg(self, svg_trees: List[etree.Element],
                            svg_sizes: List[Tuple[float, float]],
                            layout: Dict[int, np.ndarray]) -> bytes:
        """
        변환된 SVG들을 하나의 캔버스에 결합

        Args:
            svg_trees: SVG XML 트리 리스트
            svg_sizes: 각 SVG의 (width, height) 리스트
            layout: 각 SVG의 homography 행렬

        Returns:
            통합된 SVG 파일 바이트
        """
        try:
            # 1. 전체 캔버스 크기 계산
            logger.info("Calculating SVG canvas size...")
            all_corners = []

            for idx, H in layout.items():
                w, h = svg_sizes[idx]
                corners = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
                transformed_corners = cv2.perspectiveTransform(corners, H)
                all_corners.append(transformed_corners)

            all_corners = np.concatenate(all_corners, axis=0)
            [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
            [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

            canvas_width = x_max - x_min
            canvas_height = y_max - y_min

            logger.info(f"SVG canvas size: {canvas_width}x{canvas_height}")

            # 2. 새로운 SVG 캔버스 생성
            dwg = svgwrite.Drawing(size=(canvas_width, canvas_height))

            # 3. 각 SVG를 변환하여 캔버스에 추가
            logger.info("Placing SVGs on canvas...")
            for idx, H in layout.items():
                # Translation 적용 (양수 좌표로 이동)
                translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
                final_H = translation @ H

                # SVG transform 문자열 생성
                transform_str = self._homography_to_svg_transform(final_H)

                # SVG 요소들을 그룹으로 묶고 transform 적용
                g = dwg.g(transform=transform_str)

                # 원본 SVG의 모든 자식 요소 복사
                svg_tree = svg_trees[idx]
                for child in svg_tree:
                    # etree.Element를 문자열로 변환 후 추가
                    child_str = etree.tostring(child, encoding='unicode')
                    g.add(dwg.g(child_str))

                dwg.add(g)
                logger.info(f"Placed SVG {idx} with transform: {transform_str}")

            # 4. SVG를 바이트로 반환
            svg_bytes = dwg.tostring().encode('utf-8')
            logger.info("SVG stitching completed successfully!")

            return svg_bytes

        except Exception as e:
            logger.error(f"Error creating stitched SVG: {str(e)}")
            raise

    def stitch_svgs(self, svg_bytes_list: List[bytes]) -> Optional[bytes]:
        """
        여러 SVG 파일을 스티칭하여 파노라마 생성

        Args:
            svg_bytes_list: SVG 파일 바이트 리스트

        Returns:
            통합된 SVG 파일 바이트, 실패 시 None
        """
        if len(svg_bytes_list) < 2:
            logger.error("최소 2개 이상의 SVG가 필요합니다.")
            return None

        try:
            # 1. SVG 파싱 및 정보 추출
            logger.info(f"Parsing {len(svg_bytes_list)} SVG files...")
            svg_trees = []
            svg_sizes = []

            for i, svg_bytes in enumerate(svg_bytes_list):
                tree, width, height = self._parse_svg(svg_bytes)
                svg_trees.append(tree)
                svg_sizes.append((width, height))
                logger.info(f"SVG {i}: {width}x{height}")

            # 2. SVG를 고해상도 PNG로 래스터화 (매칭용)
            logger.info(f"Rasterizing SVGs to PNG (DPI={self.raster_dpi})...")
            raster_images = []

            for i, svg_bytes in enumerate(svg_bytes_list):
                img = self._svg_to_png(svg_bytes, dpi=self.raster_dpi)
                raster_images.append(img)

            # 3. 모든 이미지 쌍 매칭
            matches = self._match_all_pairs(raster_images)

            if len(matches) == 0:
                logger.error("No valid matches found between any SVG pairs")
                return None

            # 4. 매칭 그래프 구축
            graph = self._build_match_graph(len(svg_bytes_list), matches)

            # 5. 중심 이미지 찾기
            central_idx = self._find_central_image(graph)

            # 6. 각 SVG의 상대적 위치 계산
            layout = self._calculate_layout(raster_images, matches, graph, central_idx)

            if len(layout) < len(svg_bytes_list):
                logger.warning(f"Only {len(layout)}/{len(svg_bytes_list)} SVGs could be positioned.")

            # 7. 통합 SVG 생성
            return self._create_stitched_svg(svg_trees, svg_sizes, layout)

        except Exception as e:
            logger.error(f"SVG stitching failed: {str(e)}")
            return None
