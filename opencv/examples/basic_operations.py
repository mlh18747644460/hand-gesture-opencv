import cv2
import numpy as np
from typing import Optional, Tuple


def read_image(image_path: str, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    image = cv2.imread(image_path, flags)
    if image is None:
        print(f"Error: Cannot read image from {image_path}")
    return image


def save_image(image: np.ndarray, save_path: str) -> bool:
    success = cv2.imwrite(save_path, image)
    if success:
        print(f"Image saved successfully to {save_path}")
    else:
        print(f"Error: Failed to save image to {save_path}")
    return success


def display_image(window_name: str, image: np.ndarray, wait_time: int = 0) -> None:
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, image)
    cv2.waitKey(wait_time)
    if wait_time == 0:
        cv2.destroyWindow(window_name)


def display_multiple(images: dict, wait_time: int = 0) -> None:
    for name, img in images.items():
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.imshow(name, img)
    cv2.waitKey(wait_time)
    if wait_time == 0:
        cv2.destroyAllWindows()


def convert_color(image: np.ndarray, code: int) -> np.ndarray:
    return cv2.cvtColor(image, code)


def bgr_to_gray(image: np.ndarray) -> np.ndarray:
    return convert_color(image, cv2.COLOR_BGR2GRAY)


def bgr_to_hsv(image: np.ndarray) -> np.ndarray:
    return convert_color(image, cv2.COLOR_BGR2HSV)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return convert_color(image, cv2.COLOR_BGR2RGB)


def get_image_info(image: np.ndarray) -> dict:
    if len(image.shape) == 2:
        height, width = image.shape
        channels = 1
    else:
        height, width, channels = image.shape
    return {
        "width": width,
        "height": height,
        "channels": channels,
        "dtype": str(image.dtype),
        "size": image.size
    }


def resize_image(image: np.ndarray, size: Tuple[int, int], 
                 interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    return cv2.resize(image, size, interpolation=interpolation)


def resize_image_scale(image: np.ndarray, fx: float, fy: float,
                       interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    return cv2.resize(image, None, fx=fx, fy=fy, interpolation=interpolation)


def crop_image(image: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    return image[y:y+h, x:x+w]


def rotate_image(image: np.ndarray, angle: float, 
                 center: Optional[Tuple[int, int]] = None,
                 scale: float = 1.0) -> np.ndarray:
    height, width = image.shape[:2]
    if center is None:
        center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(image, rotation_matrix, (width, height))


def flip_image(image: np.ndarray, flip_code: int) -> np.ndarray:
    return cv2.flip(image, flip_code)


def split_channels(image: np.ndarray) -> Tuple[np.ndarray, ...]:
    return cv2.split(image)


def merge_channels(channels: Tuple[np.ndarray, ...]) -> np.ndarray:
    return cv2.merge(channels)


def create_sample_image(width: int = 400, height: int = 300) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.rectangle(image, (50, 50), (150, 150), (0, 0, 255), -1)
    cv2.circle(image, (250, 100), 50, (0, 255, 0), -1)
    cv2.line(image, (50, 200), (350, 250), (255, 0, 0), 3)
    cv2.putText(image, "OpenCV Sample", (80, 280), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return image


def run_basic_demo():
    print("=" * 50)
    print("基础图像操作演示")
    print("=" * 50)

    image = create_sample_image(500, 400)
    print(f"\n原始图像信息: {get_image_info(image)}")

    gray = bgr_to_gray(image)
    hsv = bgr_to_hsv(image)
    print(f"灰度图像信息: {get_image_info(gray)}")

    resized = resize_image(image, (250, 200))
    print(f"缩放后图像信息: {get_image_info(resized)}")

    scaled = resize_image_scale(image, 0.5, 0.5)
    print(f"比例缩放后图像信息: {get_image_info(scaled)}")

    cropped = crop_image(image, 50, 50, 200, 150)
    print(f"裁剪后图像信息: {get_image_info(cropped)}")

    rotated = rotate_image(image, 45)
    flipped_h = flip_image(image, 1)
    flipped_v = flip_image(image, 0)

    b, g, r = split_channels(image)
    merged = merge_channels((b, g, r))

    print("\n显示所有图像 (按任意键关闭)...")
    display_multiple({
        "Original": image,
        "Gray": gray,
        "Resized": resized,
        "Cropped": cropped,
        "Rotated 45deg": rotated,
        "Flipped Horizontal": flipped_h
    })

    save_image(image, "output_original.png")
    save_image(gray, "output_gray.png")
    print("\n基础图像操作演示完成!")


if __name__ == "__main__":
    run_basic_demo()
