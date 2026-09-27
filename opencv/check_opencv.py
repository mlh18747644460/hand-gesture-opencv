import sys
import os
import time
from datetime import datetime


def print_header(title):
    line = "=" * 60
    print(f"\n{line}")
    print(f"  {title}")
    print(line)


def print_section(title):
    print(f"\n--- {title} ---")


def check_pass(condition, test_name, detail=""):
    status = "✅ PASS" if condition else "❌ FAIL"
    msg = f"  [{status}] {test_name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    return condition


def test_environment():
    print_header("OpenCV 环境检测工具")
    print(f"检测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python 可执行文件: {sys.executable}")
    
    all_pass = True
    
    # ===== 1. Python 环境 =====
    print_section("Python 环境")
    py_version = sys.version_info
    py_ok = py_version >= (3, 8)
    all_pass &= check_pass(
        py_ok,
        "Python 版本 >= 3.8",
        f"当前版本: {py_version.major}.{py_version.minor}.{py_version.micro}"
    )
    
    # ===== 2. 核心依赖导入 =====
    print_section("核心依赖导入测试")
    
    try:
        import numpy as np
        check_pass(True, "NumPy 导入成功", f"版本: {np.__version__}")
    except ImportError as e:
        check_pass(False, "NumPy 导入成功", f"错误: {e}")
        all_pass = False
        print("\nNumPy 未安装，请运行: pip install numpy")
        return False
    
    try:
        import cv2
        cv2_ok = True
        check_pass(True, "OpenCV 导入成功", f"版本: {cv2.__version__}")
    except ImportError as e:
        cv2_ok = False
        all_pass = False
        check_pass(False, "OpenCV 导入成功", f"错误: {e}")
        print("\nOpenCV 未安装，请运行: pip install opencv-python")
        print("如需要 SIFT 等专利算法，还要安装: pip install opencv-contrib-python")
        return False
    
    # ===== 3. OpenCV 版本和构建信息 =====
    print_section("OpenCV 详细信息")
    
    print(f"  OpenCV 版本: {cv2.__version__}")
    
    try:
        build_info = cv2.getBuildInformation()
        has_cuda = "CUDA" in build_info and "YES" in build_info.split("CUDA")[1].split("\n")[0]
    except Exception:
        has_cuda = False
    
    try:
        cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
        cuda_ok = cuda_count > 0
        check_pass(cuda_ok, "CUDA 支持", f"可用设备数: {cuda_count}" if cuda_ok else "未启用")
    except (AttributeError, Exception):
        check_pass(False, "CUDA 支持", "未编译 CUDA 模块（非必须）")
    
    # ===== 4. 核心模块测试 =====
    print_section("核心模块功能测试")
    
    # 4.1 Core 模块 - 基础数组操作
    try:
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :] = (255, 0, 0)
        check_pass(True, "Core: 数组创建与赋值")
    except Exception as e:
        all_pass &= check_pass(False, "Core: 数组创建与赋值", str(e))
        all_pass = False
    
    # 4.2 ImgProc 模块 - 图像处理
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        edges = cv2.Canny(gray, 50, 150)
        check_pass(True, "ImgProc: 色彩转换/模糊/边缘检测")
    except Exception as e:
        check_pass(False, "ImgProc: 色彩转换/模糊/边缘检测", str(e))
        all_pass = False
    
    # 4.3 阈值和形态学
    try:
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=1)
        check_pass(True, "ImgProc: 阈值分割/形态学操作")
    except Exception as e:
        check_pass(False, "ImgProc: 阈值分割/形态学操作", str(e))
        all_pass = False
    
    # 4.4 几何变换
    try:
        resized = cv2.resize(img, (200, 200))
        rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        M = np.float32([[1, 0, 10], [0, 1, 20]])
        translated = cv2.warpAffine(img, M, (100, 100))
        check_pass(True, "ImgProc: 缩放/旋转/平移变换")
    except Exception as e:
        check_pass(False, "ImgProc: 缩放/旋转/平移变换", str(e))
        all_pass = False
    
    # 4.5 轮廓检测
    try:
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contoured = cv2.drawContours(img.copy(), contours, -1, (0, 255, 0), 2)
        check_pass(True, "ImgProc: 轮廓检测与绘制")
    except Exception as e:
        check_pass(False, "ImgProc: 轮廓检测与绘制", str(e))
        all_pass = False
    
    # ===== 5. Features2D 模块 - 特征检测 =====
    print_section("特征检测模块 (Features2D) 测试")
    
    # 5.1 ORB (免费可用)
    try:
        orb = cv2.ORB_create(nfeatures=100)
        test_img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(test_gray, None)
        check_pass(True, "ORB 特征检测", f"检测到 {len(kp)} 个关键点")
    except Exception as e:
        check_pass(False, "ORB 特征检测", str(e))
        all_pass = False
    
    # 5.2 FAST 角点
    try:
        fast = cv2.FastFeatureDetector_create(threshold=25)
        kp_fast = fast.detect(test_gray, None)
        check_pass(True, "FAST 角点检测", f"检测到 {len(kp_fast)} 个角点")
    except Exception as e:
        check_pass(False, "FAST 角点检测", str(e))
        all_pass = False
    
    # 5.3 SIFT (需要 contrib 模块)
    try:
        sift = cv2.SIFT_create(nfeatures=100)
        kp_sift, des_sift = sift.detectAndCompute(test_gray, None)
        check_pass(True, "SIFT 特征检测 (contrib)", f"检测到 {len(kp_sift)} 个关键点")
        sift_available = True
    except cv2.error as e:
        sift_available = False
        check_pass(False, "SIFT 特征检测 (contrib)", f"需要 opencv-contrib-python: {e}")
    except Exception as e:
        sift_available = False
        check_pass(False, "SIFT 特征检测 (contrib)", f"错误: {e}")
        all_pass = False
    
    # ===== 6. HighGUI 模块 - 图形界面 (不弹窗，仅测试API) =====
    print_section("图形界面 (HighGUI) 测试")
    
    try:
        _ = cv2.WINDOW_NORMAL
        _ = cv2.WINDOW_AUTOSIZE
        has_namedWindow = hasattr(cv2, 'namedWindow')
        has_imshow = hasattr(cv2, 'imshow')
        has_waitKey = hasattr(cv2, 'waitKey')
        gui_ok = has_namedWindow and has_imshow and has_waitKey
        check_pass(gui_ok, "HighGUI API 可用", "API存在" if gui_ok else "缺失API")
    except Exception as e:
        check_pass(False, "HighGUI API 可用", str(e))
        all_pass = False
    
    # ===== 7. 图像文件 I/O 测试 =====
    print_section("图像文件 I/O 测试")
    
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    test_file = os.path.join(output_dir, "_opencv_test_image.png")
    
    # 7.1 写文件
    try:
        write_img = np.zeros((100, 150, 3), dtype=np.uint8)
        cv2.circle(write_img, (75, 50), 30, (0, 255, 0), -1)
        cv2.putText(write_img, "OK", (55, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        write_ok = cv2.imwrite(test_file, write_img)
        check_pass(write_ok, "图像写入 (imwrite)", f"文件: {os.path.basename(test_file)}")
    except Exception as e:
        write_ok = False
        check_pass(False, "图像写入 (imwrite)", str(e))
        all_pass = False
    
    # 7.2 读文件
    read_ok = False
    if write_ok and os.path.exists(test_file):
        try:
            read_img = cv2.imread(test_file)
            read_ok = read_img is not None and read_img.shape == (100, 150, 3)
            check_pass(read_ok, "图像读取 (imread)", f"尺寸: {read_img.shape if read_ok else '失败'}")
        except Exception as e:
            check_pass(False, "图像读取 (imread)", str(e))
            all_pass = False
    else:
        check_pass(False, "图像读取 (imread)", "跳过（写入失败）")
        all_pass = False
    
    # 7.3 清理测试文件
    try:
        if os.path.exists(test_file):
            os.remove(test_file)
    except Exception:
        pass
    
    # ===== 8. 视频模块测试 (不打开摄像头) =====
    print_section("视频模块 (VideoIO) 测试")
    
    try:
        has_videocapture = hasattr(cv2, 'VideoCapture')
        has_videowriter = hasattr(cv2, 'VideoWriter')
        video_ok = has_videocapture and has_videowriter
        check_pass(video_ok, "VideoCapture/VideoWriter API 可用", "可读取视频/摄像头" if video_ok else "缺失")
    except Exception as e:
        check_pass(False, "VideoCapture/VideoWriter API 可用", str(e))
        all_pass = False
    
    # 测试 VideoWriter_fourcc
    try:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        check_pass(True, "VideoWriter_fourcc 编码可用", "XVID 编码器")
    except Exception as e:
        check_pass(False, "VideoWriter_fourcc 编码可用", str(e))
        all_pass = False
    
    # ===== 9. 性能简单测试 =====
    print_section("简单性能测试 (仅供参考)")
    
    try:
        perf_img = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)
        perf_gray = cv2.cvtColor(perf_img, cv2.COLOR_BGR2GRAY)
        
        start = time.perf_counter()
        for _ in range(10):
            _ = cv2.GaussianBlur(perf_img, (15, 15), 0)
        blur_time = (time.perf_counter() - start) / 10 * 1000
        check_pass(True, "高斯模糊性能", f"{blur_time:.1f} ms/次 (500x500)")
        
        start = time.perf_counter()
        orb_p = cv2.ORB_create(nfeatures=500)
        for _ in range(10):
            _ = orb_p.detectAndCompute(perf_gray, None)
        orb_time = (time.perf_counter() - start) / 10 * 1000
        check_pass(True, "ORB 特征检测性能", f"{orb_time:.1f} ms/次 (500x500)")
        
    except Exception as e:
        check_pass(False, "性能测试", str(e))
    
    # ===== 10. 可选依赖检查 =====
    print_section("可选依赖检查")
    
    for lib_name, import_name, pip_name in [
        ("Matplotlib", "matplotlib", "matplotlib"),
        ("Pillow (PIL)", "PIL", "Pillow"),
    ]:
        try:
            lib = __import__(import_name)
            ver = getattr(lib, "__version__", "未知")
            check_pass(True, f"{lib_name}", f"版本: {ver}")
        except ImportError:
            check_pass(False, f"{lib_name}", f"未安装 (pip install {pip_name})")
    
    # ===== 最终总结 =====
    print_header("检测结果总结")
    
    print(f"\n  Python:    {py_version.major}.{py_version.minor}.{py_version.micro}")
    print(f"  OpenCV:    {cv2.__version__}")
    print(f"  NumPy:     {np.__version__}")
    print(f"  CUDA:      {'启用' if has_cuda else '未启用'}")
    print(f"  SIFT:      {'可用' if sift_available else '需要 contrib 模块'}")
    print(f"  GUI:       {'可用' if gui_ok else 'API存在/受限'}")
    
    if all_pass:
        print(f"\n  🎉  OpenCV 环境检测全部通过！可以正常使用。")
        print(f"  现在可以运行: python main.py 启动示例演示")
    else:
        print(f"\n  ⚠️  部分检测项未通过，但核心功能大概率可用。")
        print(f"  如需使用未通过的功能，请根据上述提示安装对应依赖。")
    
    print("=" * 60 + "\n")
    
    return all_pass


if __name__ == "__main__":
    try:
        success = test_environment()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n检测已被用户中断。")
        sys.exit(1)
    except Exception as exc:
        print(f"\n\n检测过程中出现未预期错误: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(2)
