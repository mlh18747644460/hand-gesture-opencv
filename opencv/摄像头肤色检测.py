import cv2
import numpy as np
import sys
import os


def skin_ycrcb(frame, y_lo=0, y_hi=255,
               cr_lo=133, cr_hi=173,
               cb_lo=77, cb_hi=127):
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    lo = np.array([y_lo, cr_lo, cb_lo], dtype=np.uint8)
    hi = np.array([y_hi, cr_hi, cb_hi], dtype=np.uint8)
    return cv2.inRange(ycrcb, lo, hi)


def skin_hsv(frame,
             h_lo=0, h_hi=25,
             s_lo=30, s_hi=255,
             v_lo=60, v_hi=255):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lo = np.array([h_lo, s_lo, v_lo], dtype=np.uint8)
    hi = np.array([h_hi, s_hi, v_hi], dtype=np.uint8)
    mask1 = cv2.inRange(hsv, lo, hi)
    if h_lo > h_hi:
        lo2 = np.array([h_lo, s_lo, v_lo], dtype=np.uint8)
        hi2 = np.array([180, s_hi, v_hi], dtype=np.uint8)
        mask2 = cv2.inRange(hsv, lo2, hi2)
        mask1 = cv2.bitwise_or(mask1, mask2)
    return mask1


def skin_rgb(frame,
             r_lo=95, g_lo=40, b_lo=20,
             rg_min=15, rb_min=15,
             gb_max=0, rmax=220):
    b, g, r = cv2.split(frame)
    cond = (r > r_lo) & (g > g_lo) & (b > b_lo) \
           & (cv2.max(r, g) - cv2.min(r, g) > rg_min) \
           & (r - b > rb_min) \
           & (r > g) & (r > b)
    if gb_max < 0:
        cond = cond & (np.abs(g.astype(int) - b.astype(int)) < -gb_max)
    if rmax < 255:
        cond = cond & (r < rmax)
    return cond.astype(np.uint8) * 255


def skin_combo(frame, ycrcb_cfg=None, hsv_cfg=None, rgb_cfg=None):
    m1 = skin_ycrcb(frame, **(ycrcb_cfg or {}))
    m2 = skin_hsv(frame, **(hsv_cfg or {}))
    m = cv2.bitwise_and(m1, m2)
    if rgb_cfg is not None and rgb_cfg:
        m3 = skin_rgb(frame, **rgb_cfg)
        m = cv2.bitwise_and(m, m3)
    return m


def clean_mask(mask, open_k=3, close_k=7, blur_k=5):
    if open_k and open_k > 0:
        ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ko)
    if close_k and close_k > 0:
        kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kc)
    if blur_k and blur_k > 1:
        mask = cv2.GaussianBlur(mask, (blur_k | 1, blur_k | 1), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask


def draw_stats(img, mask, color=(0, 255, 0)):
    total = max(1, mask.size // 3)
    if mask.ndim == 2:
        total = mask.size
    ratio = 100.0 * cv2.countNonZero(mask) / total
    h, w = img.shape[:2]
    bar_w = int(w * 0.45)
    bar_h = 12
    x0, y0 = w - bar_w - 12, 44
    cv2.rectangle(img, (x0 - 1, y0 - 1),
                  (x0 + bar_w + 1, y0 + bar_h + 1), (0, 0, 0), -1)
    cv2.rectangle(img, (x0, y0),
                  (x0 + int(bar_w * ratio / 100), y0 + bar_h), color, -1)
    cv2.rectangle(img, (x0, y0), (x0 + bar_w, y0 + bar_h), (255, 255, 255), 1)
    cv2.putText(img, f"Skin {ratio:5.1f}%",
                (x0, y0 + bar_h + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def apply_overlay(frame, mask, mode, box_color=(0, 255, 0)):
    h, w = frame.shape[:2]
    out = frame.copy()

    if mode == 'green':
        green = np.zeros_like(frame)
        green[:] = (0, 255, 0)
        fg = cv2.bitwise_and(frame, frame, mask=mask)
        bg = cv2.bitwise_and(green, green, mask=cv2.bitwise_not(mask))
        out = cv2.add(fg, bg)
    elif mode == 'highlight':
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        fg = cv2.bitwise_and(frame, frame, mask=mask)
        dim = cv2.bitwise_and(
            cv2.addWeighted(gray3, 0.5, np.zeros_like(frame), 0.5, 0),
            gray3, mask=cv2.bitwise_not(mask))
        out = cv2.add(fg, dim)
    elif mode == 'contour':
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) > 400]
        cv2.drawContours(out, contours, -1, box_color, 2)
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            cv2.rectangle(out, (x, y), (x + bw, y + bh), box_color, 2)
            cv2.putText(out, f"P {area:.0f}", (x, max(y - 4, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1, cv2.LINE_AA)
    elif mode == 'mask':
        out = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    elif mode == 'keep':
        out = cv2.bitwise_and(frame, frame, mask=mask)
    return out


def main():
    print("=" * 60)
    print("  OpenCV 摄像头 肤色检测 演示")
    print("=" * 60)
    print("  M  = 切换算法:  YCrCb -> HSV -> RGB -> 组合(默认)")
    print("  N  = 切换显示:  highlight -> green -> contour -> mask -> keep -> original")
    print("  C  = 切换是否清理 mask（开/闭运算+模糊）")
    print("  +/-= 敏感度 ±5 （所有算法阈值同步微调）")
    print("  R  = 恢复默认参数")
    print("  S  = 保存截图")
    print("  Q / ESC = 退出")
    print("=" * 60)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[错误] 无法打开摄像头")
        return

    algo_names = ['YCrCb', 'HSV', 'RGB', 'Combo(YCrCb+HSV)']
    mode_names = ['highlight', 'green', 'contour', 'mask', 'keep', 'original']
    algo_idx = 3
    mode_idx = 0
    clean_on = True
    sens = 0
    tick_start = cv2.getTickCount()
    frames = 0
    fps = 0.0

    WNAME = "Skin Detection  (M/N/C/R +/- S/Q)"
    cv2.namedWindow(WNAME, cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[错误] 读取帧失败")
            break
        frames += 1
        frame = cv2.flip(frame, 1)

        s = sens
        ycrcb_cfg = dict(cr_lo=133 - s, cr_hi=173 + s,
                         cb_lo=77 - s, cb_hi=127 + s)
        hsv_cfg = dict(h_lo=0, h_hi=25 + s // 3,
                       s_lo=max(10, 30 - s), s_hi=255,
                       v_lo=max(30, 60 - s), v_hi=255)
        rgb_cfg = dict(r_lo=max(30, 95 - s), g_lo=max(10, 40 - s // 2),
                       b_lo=max(10, 20 - s // 2),
                       rg_min=max(5, 15 - s // 3),
                       rb_min=max(5, 15 - s // 3),
                       gb_max=-45, rmax=255)

        if algo_idx == 0:
            mask = skin_ycrcb(frame, **ycrcb_cfg)
        elif algo_idx == 1:
            mask = skin_hsv(frame, **hsv_cfg)
        elif algo_idx == 2:
            mask = skin_rgb(frame, **rgb_cfg)
        else:
            mask = skin_combo(frame, ycrcb_cfg, hsv_cfg, {})

        if clean_on:
            mask = clean_mask(mask, 3, 7, 5)

        mode = mode_names[mode_idx]
        if mode == 'original':
            disp = frame.copy()
        else:
            disp = apply_overlay(frame, mask, mode)

        algo = algo_names[algo_idx]
        h, w = disp.shape[:2]
        cv2.rectangle(disp, (0, 0), (w - 1, 34), (0, 0, 0), -1)
        cv2.putText(disp,
                    f"Algo: {algo}    View: {mode}    "
                    f"Clean:{int(clean_on)}    Sens:{sens:+d}    "
                    f"FPS:{fps:.1f}",
                    (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 255), 1, cv2.LINE_AA)
        draw_stats(disp, mask, (0, 255, 0))
        cv2.putText(disp,
                    "M:算法 N:视图 C:清理 R:复位 +/-:敏感 S:保存 Q:退出",
                    (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (220, 220, 220), 1, cv2.LINE_AA)

        cv2.imshow(WNAME, disp)
        if frames % 15 == 0:
            t = cv2.getTickCount()
            fps = 15.0 * cv2.getTickFrequency() / (t - tick_start)
            tick_start = t

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            print("退出...")
            break
        elif key == ord('m'):
            algo_idx = (algo_idx + 1) % len(algo_names)
            print(f"[算法] -> {algo_names[algo_idx]}")
        elif key == ord('n'):
            mode_idx = (mode_idx + 1) % len(mode_names)
            print(f"[视图] -> {mode_names[mode_idx]}")
        elif key == ord('c'):
            clean_on = not clean_on
            print(f"[清理] -> {'开' if clean_on else '关'}")
        elif key == ord('r'):
            sens = 0
            print("[参数] 已重置")
        elif key in (ord('+'), ord('=')):
            sens = min(50, sens + 5)
            print(f"[敏感度] Sens={sens:+d}  (更宽松)")
        elif key == ord('-'):
            sens = max(-40, sens - 5)
            print(f"[敏感度] Sens={sens:+d}  (更严格)")
        elif key == ord('s'):
            import datetime as dt
            fname = f"skin_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            save_canvas = disp
            m3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            save_canvas = np.hstack([cv2.resize(frame, (w, h)), m3, disp])
            cv2.imwrite(fname, save_canvas)
            print(f"[保存] {fname}   ({save_canvas.shape[1]}x{save_canvas.shape[0]})")

    cap.release()
    cv2.destroyAllWindows()
    print(f"已关闭摄像头，共显示 {frames} 帧。")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[已被 Ctrl+C 中断]")
        sys.exit(0)
