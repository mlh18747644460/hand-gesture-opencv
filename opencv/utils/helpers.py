import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import os
import glob


def check_opencv_version() -> str:
    return cv2.__version__


def check_cuda_support() -> bool:
    try:
        count = cv2.cuda.getCudaEnabledDeviceCount()
        return count > 0
    except Exception:
        return False


def get_output_dir(dir_name: str = "output") -> str:
    output_dir = os.path.join(os.getcwd(), dir_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


def list_images(directory: str, extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')) -> List[str]:
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(directory, f"*{ext}")))
        image_files.extend(glob.glob(os.path.join(directory, f"*{ext.upper()}")))
    return sorted(image_files)


def plot_image_matplotlib(image: np.ndarray, title: str = "", 
                          ax=None, cmap: Optional[str] = None) -> None:
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    if len(image.shape) == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(image_rgb)
    else:
        ax.imshow(image, cmap=cmap if cmap else 'gray')
    ax.set_title(title)
    ax.axis('off')


def plot_grid(images: List[np.ndarray], titles: Optional[List[str]] = None,
              cols: int = 2, figsize: Tuple[int, int] = (14, 10),
              cmap: Optional[str] = None) -> None:
    n = len(images)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if n > 1 else [axes]
    
    for i, (img, ax) in enumerate(zip(images, axes)):
        title = titles[i] if titles and i < len(titles) else f"Image {i+1}"
        plot_image_matplotlib(img, title, ax, cmap)
    
    for i in range(n, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_histogram(image: np.ndarray, title: str = "Histogram",
                   mask: Optional[np.ndarray] = None) -> None:
    if len(image.shape) == 3:
        colors = ('b', 'g', 'r')
        for i, color in enumerate(colors):
            hist = cv2.calcHist([image], [i], mask, [256], [0, 256])
            plt.plot(hist, color=color)
        plt.legend(['Blue', 'Green', 'Red'])
    else:
        hist = cv2.calcHist([image], [0], mask, [256], [0, 256])
        plt.plot(hist, color='gray')
    plt.title(title)
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.xlim([0, 256])
    plt.grid(True, alpha=0.3)
    plt.show()


def add_text_watermark(image: np.ndarray, text: str,
                        position: Tuple[int, int] = (10, 30),
                        font_scale: float = 1.0,
                        color: Tuple[int, int, int] = (255, 255, 255),
                        thickness: int = 2) -> np.ndarray:
    result = image.copy()
    cv2.putText(result, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)
    return result


def create_comparison(images: List[np.ndarray], labels: List[str],
                       spacing: int = 20, bg_color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    assert len(images) == len(labels), "Images and labels count must match"
    
    h = max(img.shape[0] for img in images)
    w = max(img.shape[1] for img in images)
    label_height = 40
    
    n = len(images)
    total_w = n * w + (n + 1) * spacing
    total_h = h + label_height + 2 * spacing
    
    canvas = np.full((total_h, total_w, 3), bg_color, dtype=np.uint8)
    
    for i, (img, label) in enumerate(zip(images, labels)):
        x = spacing + i * (w + spacing)
        y = spacing
        
        if len(img.shape) == 2:
            img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_color = img
        
        img_h, img_w = img_color.shape[:2]
        y_offset = (h - img_h) // 2
        x_offset = (w - img_w) // 2
        
        canvas[y + y_offset:y + y_offset + img_h,
               x + x_offset:x + x_offset + img_w] = img_color
        
        label_pos = (x + w // 2 - 40, y + h + label_height - 10)
        cv2.putText(canvas, label, label_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2, cv2.LINE_AA)
    
    return canvas


def print_progress(current: int, total: int, prefix: str = "Progress:") -> None:
    percent = 100 * current / total
    bar_length = 50
    filled = int(bar_length * current / total)
    bar = '█' * filled + '░' * (bar_length - filled)
    print(f"\r{prefix} |{bar}| {percent:.1f}% ({current}/{total})", end='', flush=True)
    if current == total:
        print()


class ImageBatchProcessor:
    def __init__(self, input_dir: str, output_dir: str = "output"):
        self.input_dir = input_dir
        self.output_dir = get_output_dir(output_dir)
        self.image_paths = list_images(input_dir)
    
    def process(self, process_func, save_prefix: str = "processed_") -> List[str]:
        processed_paths = []
        n = len(self.image_paths)
        
        for i, img_path in enumerate(self.image_paths, 1):
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            processed = process_func(img)
            
            filename = os.path.basename(img_path)
            name, ext = os.path.splitext(filename)
            save_path = os.path.join(self.output_dir, f"{save_prefix}{name}{ext}")
            cv2.imwrite(save_path, processed)
            processed_paths.append(save_path)
            
            print_progress(i, n, "Processing:")
        
        return processed_paths


def print_environment_info():
    print("=" * 50)
    print("OpenCV 开发环境信息")
    print("=" * 50)
    print(f"OpenCV 版本: {check_opencv_version()}")
    print(f"NumPy 版本: {np.__version__}")
    print(f"CUDA 支持: {'是' if check_cuda_support() else '否'}")
    print(f"工作目录: {os.getcwd()}")
    print(f"输出目录: {get_output_dir()}")
    print("=" * 50)
