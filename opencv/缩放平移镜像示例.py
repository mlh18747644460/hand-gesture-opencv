import cv2
import numpy as np
import os
import sys


def resize_image(image, scale_x, scale_y=None):
    if scale_y is None:
        scale_y = scale_x
    return cv2.resize(image, None, fx=scale_x, fy=scale_y,
                      interpolation=cv2.INTER_LINEAR)


def translate_image(image, tx, ty):
    h, w = image.shape[:2]
    M = np.float32([[1, 0, tx],
                    [0, 1, ty]])
    return cv2.warpAffine(image, M, (w, h),
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(0, 0, 0))


def flip_image(image, mode):
    if mode in ('h', 'horizontal', 'left_right', '左右'):
        return cv2.flip(image, 1)
    elif mode in ('v', 'vertical', 'up_down', '上下'):
        return cv2.flip(image, 0)
    elif mode in ('b', 'both', 'hv', '双向', '180'):
        return cv2.flip(image, -1)
    else:
        return image.copy()


def label(img, text, color=(0, 255, 255)):
    out = img.copy()
    h, w = out.shape[:2]
    cv2.rectangle(out, (0, 0), (w - 1, 28), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    return out


def demo_static(image_path=None):
    if image_path and os.path.isfile(image_path):
        src = cv2.imread(image_path)
        if src is None:
            print(f"[错误] 无法读取图片: {image_path}")
            return
    else:
        src = np.zeros((320, 480, 3), dtype=np.uint8)
        src[:] = (40, 40, 40)
        cv2.circle(src, (140, 160), 70, (0, 120, 255), -1)
        cv2.rectangle(src, (260, 80), (420, 240), (0, 200, 100), -1)
        cv2.putText(src, "A", (115, 185),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 5)
        cv2.putText(src, "B", (305, 195),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 5)

    h, w = src.shape[:2]
    if max(h, w) > 500:
        s = 500 / max(h, w)
        src = resize_image(src, s)
    h, w = src.shape[:2]

    r_small   = resize_image(src, 0.6)
    r_large   = resize_image(src, 1.4)
    t_rightup = translate_image(src,  80, -50)
    t_leftdn  = translate_image(src, -70,  60)
    f_h       = flip_image(src, 'h')
    f_v       = flip_image(src, 'v')
    f_b       = flip_image(src, 'b')

    combo = src.copy()
    combo = resize_image(combo, 0.8)
    combo = translate_image(combo, 30, 20)
    combo = flip_image(combo, 'h')

    def fit(img):
        ih, iw = img.shape[:2]
        if ih != h or iw != w:
            return cv2.resize(img, (w, h))
        return img

    items = [
        label(src,        "1. Original 原图"),
        label(fit(r_small),   "2. Resize x0.6 缩放缩小"),
        label(fit(r_large),   "3. Resize x1.4 缩放放大"),
        label(t_rightup, "4. Translate (+80,-50) 右移上移"),
        label(t_leftdn,  "5. Translate (-70,+60) 左移下移"),
        label(f_h,       "6. Flip H 左右镜像"),
        label(f_v,       "7. Flip V 上下镜像"),
        label(f_b,       "8. Flip B 双向镜像(180)"),
        label(fit(combo),    "9. Combo 缩放+平移+左右镜像"),
    ]

    cols = 3
    rows = (len(items) + cols - 1) // cols
    pad = 18
    canvas_w = cols * w + (cols + 1) * pad
    canvas_h = rows * h + (rows + 1) * pad
    canvas = np.full((canvas_h, canvas_w, 3), 245, dtype=np.uint8)

    for i, it in enumerate(items):
        r, c = divmod(i, cols)
        x = pad + c * (w + pad)
        y = pad + r * (h + pad)
        canvas[y:y + h, x:x + w] = it
        cv2.rectangle(canvas, (x, y), (x + w - 1, y + h - 1), (80, 120, 200), 2)

    out_name = "缩放平移镜像示例.png"
    cv2.imwrite(out_name, canvas)
    print(f"[OK] 对比图已保存: {os.path.abspath(out_name)}  ({canvas.shape[1]}x{canvas.shape[0]})")

    cv2.imshow("缩放 / 平移 / 镜像 示例 (按任意键关闭)", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def demo_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[错误] 无法打开摄像头")
        return

    print("\n摄像头实时演示已启动：")
    print("  [+]/[-]  缩放  ±0.1")
    print("  W/A/X/D  平移  上/左/下/右  20px")
    print("  H        左右镜像 开/关")
    print("  V        上下镜像 开/关")
    print("  0        重置全部变换")
    print("  S        保存截图")
    print("  Q / ESC  退出\n")

    scale = 1.0
    tx, ty = 0, 0
    flip_h, flip_v = False, False
    frames = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1

        img = resize_image(frame, scale)
        img = translate_image(img, tx, ty)
        if flip_h:
            img = flip_image(img, 'h')
        if flip_v:
            img = flip_image(img, 'v')

        h, w = img.shape[:2]
        hud = (f"Scale:{scale:+.2f}  Move:({tx:+d},{ty:+d})  "
               f"FlipH:{int(flip_h)}  FlipV:{int(flip_v)}")
        cv2.rectangle(img, (0, 0), (w - 1, 30), (0, 0, 0), -1)
        cv2.putText(img, hud, (8, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        cv2.putText(img, "+/-:缩放  WAXD:平移  H/V:镜像  0:重置  S:保存  Q:退出",
                    (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)

        cv2.imshow("Camera 缩放/平移/镜像 实时演示", img)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:
            print("退出...")
            break
        elif key in (ord('+'), ord('=')):
            scale = round(min(scale + 0.1, 3.0), 2)
            print(f"  -> 缩放 = {scale:.2f}")
        elif key == ord('-'):
            scale = round(max(scale - 0.1, 0.3), 2)
            print(f"  -> 缩放 = {scale:.2f}")
        elif key == ord('w'):
            ty -= 20;  print(f"  -> 平移 = ({tx}, {ty})")
        elif key == ord('x'):
            ty += 20;  print(f"  -> 平移 = ({tx}, {ty})")
        elif key == ord('a'):
            tx -= 20;  print(f"  -> 平移 = ({tx}, {ty})")
        elif key == ord('d'):
            tx += 20;  print(f"  -> 平移 = ({tx}, {ty})")
        elif key == ord('h'):
            flip_h = not flip_h;  print(f"  -> 左右镜像 = {'开' if flip_h else '关'}")
        elif key == ord('v'):
            flip_v = not flip_v;  print(f"  -> 上下镜像 = {'开' if flip_v else '关'}")
        elif key == ord('0'):
            scale = 1.0; tx = ty = 0; flip_h = flip_v = False
            print("  -> 全部重置")
        elif key == ord('s'):
            import datetime as _dt
            fname = f"cam_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(fname, img)
            print(f"  -> 截图已保存: {fname}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"摄像头已关闭，共显示 {frames} 帧。")


def main():
    if len(sys.argv) < 2:
        print("=" * 58)
        print("  OpenCV  缩放 / 平移 / 镜像  三合一示例")
        print("=" * 58)
        print("  用法:")
        print("    python 缩放平移镜像示例.py camera          摄像头实时演示")
        print("    python 缩放平移镜像示例.py <图片路径>      静态图片演示")
        print("    python 缩放平移镜像示例.py                 菜单选择")
        print("=" * 58)
        print()
        print(" [1] 摄像头实时演示")
        print(" [2] 静态图片演示（使用内置样例图）")
        print(" [0] 退出")
        c = input("请选择: ").strip()
        if c == '1':
            demo_camera()
        elif c == '2':
            demo_static()
        else:
            print("已退出。")
        return

    arg = sys.argv[1]
    if arg.lower() in ('camera', 'cam', '0'):
        demo_camera()
    elif os.path.isfile(arg):
        demo_static(arg)
    else:
        print(f"[错误] 不识别的参数或文件不存在: {arg}")


if __name__ == "__main__":
    main()
