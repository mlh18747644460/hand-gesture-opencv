import cv2
import numpy as np
from typing import Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from examples.basic_operations import (
    create_sample_image, display_multiple, save_image, 
    bgr_to_gray, display_image
)


def blur_average(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    return cv2.blur(image, (kernel_size, kernel_size))


def blur_gaussian(image: np.ndarray, kernel_size: int = 5, sigma_x: float = 0) -> np.ndarray:
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma_x)


def blur_median(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    return cv2.medianBlur(image, kernel_size)


def blur_bilateral(image: np.ndarray, d: int = 9, sigma_color: float = 75,
                   sigma_space: float = 75) -> np.ndarray:
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def sharpen_simple(image: np.ndarray) -> np.ndarray:
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]], dtype=np.float32)
    return cv2.filter2D(image, -1, kernel)


def sharpen_unsharp_mask(image: np.ndarray, sigma: float = 1.0,
                         amount: float = 1.5) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    sharpened = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
    return sharpened


def sharpen_laplacian(image: np.ndarray) -> np.ndarray:
    kernel = np.array([[0, -1,  0],
                       [-1, 5, -1],
                       [0, -1,  0]], dtype=np.float32)
    return cv2.filter2D(image, -1, kernel)


def equalize_histogram_gray(gray_image: np.ndarray) -> np.ndarray:
    return cv2.equalizeHist(gray_image)


def equalize_histogram_color(image: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def clahe_gray(gray_image: np.ndarray, clip_limit: float = 2.0,
               tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray_image)


def clahe_color(image: np.ndarray, clip_limit: float = 2.0,
                tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def adjust_brightness_contrast(image: np.ndarray, alpha: float = 1.0,
                                beta: int = 0) -> np.ndarray:
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def adjust_gamma(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def add_noise_salt_pepper(image: np.ndarray, amount: float = 0.01) -> np.ndarray:
    noisy = np.copy(image)
    num_salt = np.ceil(amount * image.size * 0.5).astype(int)
    num_pepper = np.ceil(amount * image.size * 0.5).astype(int)
    
    coords_salt = [np.random.randint(0, i - 1, num_salt) for i in image.shape[:2]]
    noisy[coords_salt[0], coords_salt[1]] = 255
    
    coords_pepper = [np.random.randint(0, i - 1, num_pepper) for i in image.shape[:2]]
    noisy[coords_pepper[0], coords_pepper[1]] = 0
    
    return noisy


def threshold_simple(gray_image: np.ndarray, threshold: int = 127,
                      max_value: int = 255) -> Tuple[float, np.ndarray]:
    return cv2.threshold(gray_image, threshold, max_value, cv2.THRESH_BINARY)


def threshold_otsu(gray_image: np.ndarray, max_value: int = 255) -> Tuple[float, np.ndarray]:
    return cv2.threshold(gray_image, 0, max_value, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


def threshold_adaptive(gray_image: np.ndarray, max_value: int = 255,
                        block_size: int = 11, c: int = 2) -> np.ndarray:
    return cv2.adaptiveThreshold(gray_image, max_value,
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, block_size, c)


def edge_canny(image: np.ndarray, threshold1: float = 100,
               threshold2: float = 200) -> np.ndarray:
    gray = bgr_to_gray(image) if len(image.shape) == 3 else image
    return cv2.Canny(gray, threshold1, threshold2)


def edge_sobel(gray_image: np.ndarray, dx: int = 1, dy: int = 1,
               ksize: int = 3) -> np.ndarray:
    sobel = cv2.Sobel(gray_image, cv2.CV_64F, dx, dy, ksize=ksize)
    sobel_abs = np.absolute(sobel)
    return np.uint8(sobel_abs / sobel_abs.max() * 255) if sobel_abs.max() > 0 else np.uint8(sobel_abs)


def morphology_dilate(image: np.ndarray, kernel_size: int = 5,
                      iterations: int = 1) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(image, kernel, iterations=iterations)


def morphology_erode(image: np.ndarray, kernel_size: int = 5,
                     iterations: int = 1) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(image, kernel, iterations=iterations)


def morphology_open(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


def morphology_close(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)


def morphology_gradient(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)


def run_filter_demo():
    print("=" * 50)
    print("图像滤波与增强演示")
    print("=" * 50)

    original = create_sample_image(600, 450)
    gray = bgr_to_gray(original)

    print("\n--- 图像模糊演示 ---")
    avg_blur = blur_average(original, 9)
    gauss_blur = blur_gaussian(original, 9, 1.5)
    median_blur = blur_median(original, 9)
    bilateral_blur = blur_bilateral(original, 9, 75, 75)

    print("完成: 均值模糊、高斯模糊、中值模糊、双边模糊")

    print("\n--- 图像锐化演示 ---")
    sharp_simple = sharpen_simple(original)
    sharp_usm = sharpen_unsharp_mask(original, 1.0, 1.5)
    sharp_laplacian = sharpen_laplacian(original)
    print("完成: 简单锐化、非锐化掩膜、拉普拉斯锐化")

    print("\n--- 直方图均衡化演示 ---")
    low_contrast = adjust_brightness_contrast(gray, 0.5, 50)
    hist_eq = equalize_histogram_gray(low_contrast)
    clahe_eq = clahe_gray(low_contrast, 3.0)
    color_eq = equalize_histogram_color(original)
    print("完成: 全局直方图均衡化、CLAHE自适应均衡化、彩色直方图均衡化")

    print("\n--- 亮度对比度与伽马校正演示 ---")
    bright = adjust_brightness_contrast(original, 1.2, 30)
    dark = adjust_brightness_contrast(original, 0.8, -20)
    high_contrast = adjust_brightness_contrast(original, 1.5, 0)
    gamma_low = adjust_gamma(original, 0.5)
    gamma_high = adjust_gamma(original, 2.0)
    print("完成: 亮度/对比度调整、伽马校正")

    print("\n--- 噪声与去噪演示 ---")
    np.random.seed(42)
    noisy = add_noise_salt_pepper(original, 0.02)
    denoised_median = blur_median(noisy, 5)
    denoised_bilateral = blur_bilateral(noisy, 9, 75, 75)
    print("完成: 椒盐噪声添加、中值去噪、双边去噪")

    print("\n--- 阈值分割演示 ---")
    _, thresh_simple = threshold_simple(gray, 127)
    _, thresh_otsu = threshold_otsu(gray)
    thresh_adaptive = threshold_adaptive(gray)
    print("完成: 简单阈值、Otsu阈值、自适应阈值")

    print("\n--- 边缘检测演示 ---")
    edges_canny = edge_canny(original, 50, 150)
    edges_sobel_x = edge_sobel(gray, 1, 0)
    edges_sobel_y = edge_sobel(gray, 0, 1)
    edges_sobel = cv2.addWeighted(edges_sobel_x, 0.5, edges_sobel_y, 0.5, 0)
    print("完成: Canny边缘检测、Sobel边缘检测")

    print("\n--- 形态学操作演示 ---")
    _, binary_img = threshold_otsu(gray)
    dilated = morphology_dilate(binary_img, 5)
    eroded = morphology_erode(binary_img, 5)
    opened = morphology_open(binary_img, 5)
    closed = morphology_close(binary_img, 5)
    gradient = morphology_gradient(binary_img, 5)
    print("完成: 膨胀、腐蚀、开运算、闭运算、形态学梯度")

    print("\n显示部分结果 (按任意键关闭)...")
    display_multiple({
        "1. Original": original,
        "2. Noisy": noisy,
        "3. Denoised (Median)": denoised_median,
        "4. Sharpened (USM)": sharp_usm,
        "5. Histogram Equalized": hist_eq,
        "6. CLAHE": clahe_eq,
        "7. Canny Edges": edges_canny,
        "8. Morphology Gradient": gradient
    })

    print("\n保存部分结果...")
    save_image(gauss_blur, "output_gaussian_blur.png")
    save_image(sharp_usm, "output_sharpened.png")
    save_image(hist_eq, "output_histogram_equalized.png")
    save_image(edges_canny, "output_canny_edges.png")
    print("\n图像滤波与增强演示完成!")


if __name__ == "__main__":
    run_filter_demo()
