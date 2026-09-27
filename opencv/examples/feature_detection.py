import cv2
import numpy as np
from typing import List, Tuple, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from examples.basic_operations import (
    create_sample_image, display_multiple, save_image,
    bgr_to_gray, display_image, crop_image
)


def create_feature_image_1(width: int = 600, height: int = 450) -> np.ndarray:
    image = np.full((height, width, 3), 240, dtype=np.uint8)
    cv2.rectangle(image, (50, 50), (200, 200), (50, 50, 200), -1)
    cv2.circle(image, (400, 120), 70, (200, 100, 50), -1)
    cv2.ellipse(image, (300, 320), (120, 80), 30, 0, 360, (80, 180, 80), -1)
    for i in range(8):
        for j in range(8):
            color = (i * 25, j * 25, (i + j) * 15)
            cv2.rectangle(image, (420 + j * 20, 230 + i * 20),
                         (438 + j * 20, 248 + i * 20), color, -1)
    return image


def create_feature_image_2(width: int = 600, height: int = 450) -> np.ndarray:
    image = np.full((height, width, 3), 230, dtype=np.uint8)
    cv2.rectangle(image, (350, 200), (500, 350), (50, 50, 200), -1)
    cv2.circle(image, (150, 300), 60, (200, 100, 50), -1)
    cv2.ellipse(image, (450, 100), (100, 60), -20, 0, 360, (80, 180, 80), -1)
    for i in range(6):
        for j in range(6):
            color = (j * 30, (i + j) * 18, i * 30)
            cv2.rectangle(image, (30 + j * 25, 30 + i * 25),
                         (50 + j * 25, 50 + i * 25), color, -1)
    cv2.line(image, (200, 150), (400, 400), (100, 100, 255), 4)
    return image


def detect_sift(gray_image: np.ndarray, nfeatures: int = 0) -> Tuple:
    sift = cv2.SIFT_create(nfeatures=nfeatures)
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)
    return keypoints, descriptors


def detect_orb(gray_image: np.ndarray, nfeatures: int = 500) -> Tuple:
    orb = cv2.ORB_create(nfeatures=nfeatures)
    keypoints, descriptors = orb.detectAndCompute(gray_image, None)
    return keypoints, descriptors


def detect_fast(gray_image: np.ndarray, threshold: int = 25) -> List:
    fast = cv2.FastFeatureDetector_create(threshold=threshold)
    keypoints = fast.detect(gray_image, None)
    return keypoints


def detect_harris_corners(gray_image: np.ndarray, block_size: int = 2,
                           ksize: int = 3, k: float = 0.04,
                           threshold_ratio: float = 0.01) -> np.ndarray:
    gray_float = np.float32(gray_image)
    dst = cv2.cornerHarris(gray_float, block_size, ksize, k)
    dst = cv2.dilate(dst, None)
    mask = dst > threshold_ratio * dst.max()
    return mask


def detect_good_features(gray_image: np.ndarray, max_corners: int = 100,
                          quality_level: float = 0.01,
                          min_distance: int = 10) -> np.ndarray:
    corners = cv2.goodFeaturesToTrack(gray_image, max_corners, quality_level, min_distance)
    return corners


def draw_keypoints(image: np.ndarray, keypoints: List,
                   color: Tuple = (0, 255, 0)) -> np.ndarray:
    return cv2.drawKeypoints(image.copy(), keypoints, None,
                              color=color,
                              flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


def draw_harris_corners(image: np.ndarray, corner_mask: np.ndarray,
                        color: Tuple = (0, 0, 255)) -> np.ndarray:
    result = image.copy()
    result[corner_mask] = color
    return result


def draw_good_features(image: np.ndarray, corners: np.ndarray,
                       color: Tuple = (0, 255, 0),
                       radius: int = 3) -> np.ndarray:
    result = image.copy()
    if corners is not None:
        corners = np.int32(corners)
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(result, (x, y), radius, color, -1)
    return result


def match_knn_bf(desc1: np.ndarray, desc2: np.ndarray,
                 k: int = 2, norm_type: int = cv2.NORM_L2) -> List:
    bf = cv2.BFMatcher(norm_type)
    matches = bf.knnMatch(desc1, desc2, k=k)
    return matches


def match_bf(desc1: np.ndarray, desc2: np.ndarray,
             norm_type: int = cv2.NORM_HAMMING, cross_check: bool = True) -> List:
    bf = cv2.BFMatcher(norm_type, crossCheck=cross_check)
    matches = bf.match(desc1, desc2)
    return sorted(matches, key=lambda x: x.distance)


def filter_matches_lowe(knn_matches: List, ratio: float = 0.75) -> List:
    good_matches = []
    for match_pair in knn_matches:
        if len(match_pair) >= 2:
            m, n = match_pair
            if m.distance < ratio * n.distance:
                good_matches.append(m)
    return good_matches


def draw_matches(img1: np.ndarray, kp1: List, img2: np.ndarray, kp2: List,
                 matches: List, match_count: Optional[int] = None) -> np.ndarray:
    if match_count is None:
        match_count = len(matches)
    return cv2.drawMatches(img1, kp1, img2, kp2,
                            matches[:match_count], None,
                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


def template_matching(image: np.ndarray, template: np.ndarray,
                      method: int = cv2.TM_CCOEFF_NORMED) -> Tuple[float, Tuple[int, int]]:
    result = cv2.matchTemplate(image, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        return min_val, min_loc
    else:
        return max_val, max_loc


def multi_template_matching(image: np.ndarray, template: np.ndarray,
                            threshold: float = 0.8,
                            method: int = cv2.TM_CCOEFF_NORMED) -> List[Tuple[int, int]]:
    result = cv2.matchTemplate(image, template, method)
    locations = np.where(result >= threshold) if method not in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED] \
        else np.where(result <= 1 - threshold)
    positions = list(zip(*locations[::-1]))
    return positions


def draw_template_matches(image: np.ndarray, template: np.ndarray,
                          positions: List[Tuple[int, int]]) -> np.ndarray:
    result = image.copy()
    h, w = template.shape[:2]
    for pt in positions:
        cv2.rectangle(result, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
    return result


def find_homography(kp1: List, kp2: List, good_matches: List,
                     ransac_thresh: float = 5.0) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if len(good_matches) < 4:
        return None, None
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
    return M, mask


def stitch_images(img1: np.ndarray, img2: np.ndarray,
                   H: Optional[np.ndarray] = None,
                   kp1: Optional[List] = None, kp2: Optional[List] = None,
                   matches: Optional[List] = None) -> np.ndarray:
    if H is None and all(v is not None for v in [kp1, kp2, matches]):
        H, _ = find_homography(kp1, kp2, matches)
    if H is None:
        raise ValueError("无法计算单应性矩阵，匹配点不足")
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    result = cv2.warpPerspective(img1, H, (w1 + w2, max(h1, h2)))
    result[0:h2, 0:w2] = img2
    return result


def detect_contours(gray_image: np.ndarray,
                     mode: int = cv2.RETR_EXTERNAL,
                     method: int = cv2.CHAIN_APPROX_SIMPLE) -> Tuple[List, np.ndarray]:
    contours, hierarchy = cv2.findContours(gray_image, mode, method)
    return contours, hierarchy


def draw_contours(image: np.ndarray, contours: List,
                  color: Tuple = (0, 255, 0), thickness: int = 2) -> np.ndarray:
    result = image.copy()
    cv2.drawContours(result, contours, -1, color, thickness)
    return result


def contour_properties(contour: np.ndarray) -> dict:
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    x, y, w, h = cv2.boundingRect(contour)
    rect_area = w * h
    extent = float(area) / rect_area if rect_area > 0 else 0
    (cx, cy), radius = cv2.minEnclosingCircle(contour)
    M = cv2.moments(contour)
    center_x = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
    center_y = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0
    return {
        "area": area,
        "perimeter": perimeter,
        "bounding_rect": (x, y, w, h),
        "extent": extent,
        "enclosing_circle": (int(cx), int(cy), int(radius)),
        "centroid": (center_x, center_y)
    }


def hough_lines(gray_image: np.ndarray, rho: float = 1, theta: float = np.pi / 180,
                threshold: int = 100) -> np.ndarray:
    edges = cv2.Canny(gray_image, 50, 150)
    return cv2.HoughLines(edges, rho, theta, threshold)


def draw_hough_lines(image: np.ndarray, lines: np.ndarray,
                     color: Tuple = (0, 0, 255), thickness: int = 2) -> np.ndarray:
    result = image.copy()
    if lines is not None:
        for rho_theta in lines:
            rho, theta = rho_theta[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(result, (x1, y1), (x2, y2), color, thickness)
    return result


def hough_circles(gray_image: np.ndarray, dp: float = 1.2, min_dist: float = 50,
                  param1: float = 100, param2: float = 30,
                  min_radius: int = 10, max_radius: int = 100) -> np.ndarray:
    return cv2.HoughCircles(gray_image, cv2.HOUGH_GRADIENT, dp, min_dist,
                             param1=param1, param2=param2,
                             minRadius=min_radius, maxRadius=max_radius)


def draw_hough_circles(image: np.ndarray, circles: np.ndarray,
                       color: Tuple = (0, 255, 0), thickness: int = 2) -> np.ndarray:
    result = image.copy()
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            cv2.circle(result, (circle[0], circle[1]), circle[2], color, thickness)
            cv2.circle(result, (circle[0], circle[1]), 2, (0, 0, 255), 3)
    return result


def run_feature_demo():
    print("=" * 50)
    print("特征检测与匹配演示")
    print("=" * 50)

    img1 = create_feature_image_1()
    img2 = create_feature_image_2()
    gray1 = bgr_to_gray(img1)
    gray2 = bgr_to_gray(img2)

    print("\n--- SIFT 特征检测 ---")
    kp1_sift, desc1_sift = detect_sift(gray1)
    kp2_sift, desc2_sift = detect_sift(gray2)
    img1_sift = draw_keypoints(img1, kp1_sift, (0, 255, 0))
    img2_sift = draw_keypoints(img2, kp2_sift, (0, 255, 0))
    print(f"图像1 SIFT关键点数量: {len(kp1_sift)}")
    print(f"图像2 SIFT关键点数量: {len(kp2_sift)}")

    print("\n--- SIFT 特征匹配 (KNN + Lowe's ratio test) ---")
    knn_matches_sift = match_knn_bf(desc1_sift, desc2_sift, k=2)
    good_sift = filter_matches_lowe(knn_matches_sift, ratio=0.7)
    sift_matched = draw_matches(img1, kp1_sift, img2, kp2_sift, good_sift, 50)
    print(f"良好匹配数量: {len(good_sift)}")

    print("\n--- ORB 特征检测与匹配 ---")
    kp1_orb, desc1_orb = detect_orb(gray1, 800)
    kp2_orb, desc2_orb = detect_orb(gray2, 800)
    img1_orb = draw_keypoints(img1, kp1_orb, (255, 0, 0))
    orb_matches = match_bf(desc1_orb, desc2_orb, cv2.NORM_HAMMING, True)
    orb_matched = draw_matches(img1, kp1_orb, img2, kp2_orb, orb_matches, 50)
    print(f"图像1 ORB关键点数量: {len(kp1_orb)}")
    print(f"ORB匹配数量 (前50显示): {len(orb_matches)}")

    print("\n--- FAST 角点检测 ---")
    kp_fast = detect_fast(gray1, 30)
    img_fast = draw_keypoints(img1, kp_fast, (255, 255, 0))
    print(f"FAST角点数量: {len(kp_fast)}")

    print("\n--- Harris 角点检测 ---")
    harris_mask = detect_harris_corners(gray1, 2, 3, 0.04, 0.01)
    img_harris = draw_harris_corners(img1, harris_mask)

    print("\n--- Shi-Tomasi 角点检测 ---")
    good_features = detect_good_features(gray1, 150, 0.01, 10)
    img_shitomasi = draw_good_features(img1, good_features, (0, 255, 255), 4)
    print(f"Shi-Tomasi角点数量: {len(good_features) if good_features is not None else 0}")

    print("\n--- 模板匹配 ---")
    template = crop_image(img1, 50, 50, 150, 150)
    match_val, match_loc = template_matching(img2, template, cv2.TM_CCOEFF_NORMED)
    template_result = img2.copy()
    th, tw = template.shape[:2]
    cv2.rectangle(template_result, match_loc,
                  (match_loc[0] + tw, match_loc[1] + th), (0, 0, 255), 3)
    print(f"模板匹配最佳相似度: {match_val:.4f}")
    print(f"模板匹配位置: {match_loc}")

    print("\n--- 轮廓检测 ---")
    _, binary1 = cv2.threshold(gray1, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = detect_contours(binary1)
    img_contours = draw_contours(img1, contours, (0, 255, 0), 2)
    print(f"检测到的轮廓数量: {len(contours)}")
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        props = contour_properties(largest_contour)
        print(f"最大轮廓属性: 面积={props['area']:.1f}, "
              f"周长={props['perimeter']:.1f}, 质心={props['centroid']}")

    print("\n--- 霍夫直线检测 ---")
    hough_lines_img = np.full((400, 500, 3), 255, dtype=np.uint8)
    cv2.line(hough_lines_img, (50, 50), (450, 50), (0, 0, 0), 3)
    cv2.line(hough_lines_img, (50, 200), (450, 350), (0, 0, 0), 3)
    cv2.line(hough_lines_img, (100, 350), (400, 100), (0, 0, 0), 3)
    hough_gray = bgr_to_gray(hough_lines_img)
    lines = hough_lines(hough_gray, 1, np.pi / 180, 150)
    hough_result = draw_hough_lines(hough_lines_img, lines)
    print(f"霍夫检测到直线数量: {len(lines) if lines is not None else 0}")

    print("\n--- 霍夫圆检测 ---")
    hough_circles_img = np.full((400, 500, 3), 255, dtype=np.uint8)
    cv2.circle(hough_circles_img, (120, 120), 50, (0, 0, 0), -1)
    cv2.circle(hough_circles_img, (350, 150), 70, (0, 0, 0), -1)
    cv2.circle(hough_circles_img, (250, 300), 40, (0, 0, 0), -1)
    hough_circ_gray = bgr_to_gray(hough_circles_img)
    circles = hough_circles(hough_circ_gray, 1.2, 60, 100, 25, 20, 100)
    circles_result = draw_hough_circles(hough_circles_img, circles)
    print(f"霍夫检测到圆数量: {len(circles[0]) if circles is not None else 0}")

    print("\n显示部分结果 (按任意键关闭)...")
    display_multiple({
        "1. SIFT Keypoints (Img1)": img1_sift,
        "2. SIFT Matches": sift_matched,
        "3. ORB Matches": orb_matched,
        "4. FAST Corners": img_fast,
        "5. Harris Corners": img_harris,
        "6. Shi-Tomasi Corners": img_shitomasi,
        "7. Template Matching": template_result,
        "8. Contours": img_contours
    })

    print("\n保存部分结果...")
    save_image(sift_matched, "output_sift_matches.png")
    save_image(orb_matched, "output_orb_matches.png")
    save_image(img_contours, "output_contours.png")
    save_image(hough_result, "output_hough_lines.png")
    save_image(circles_result, "output_hough_circles.png")
    print("\n特征检测与匹配演示完成!")


if __name__ == "__main__":
    run_feature_demo()
