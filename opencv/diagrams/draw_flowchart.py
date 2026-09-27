# -*- coding: utf-8 -*-
"""
系统流程图绘制器 —— 为项目报告生成《手势识别PPT控制系统流程图.png》
输出文件：e:\opencv\diagrams\系统流程图.png
"""
import os
import sys
from PIL import Image, ImageDraw, ImageFont

OUT_DIR = r"e:\opencv\diagrams"
OUT_FILE = os.path.join(OUT_DIR, "系统流程图.png")
os.makedirs(OUT_DIR, exist_ok=True)

def find_font(size: int):
    candidates = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyhbd.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()

F14 = find_font(14)
F16 = find_font(16)
F18 = find_font(18)
F22B = find_font(22)
F28B = find_font(28)

CW, CH = 2800, 2000
BG = (248, 250, 252)
WHITE = (255, 255, 255)
BLACK = (30, 41, 59)
GRAY = (148, 163, 184)
DARK = (71, 85, 105)

C_COLLECT  = (219, 234, 254)  # 浅蓝
C_RECOG    = (220, 252, 231)  # 浅绿
C_CTRL     = (254, 243, 199)  # 浅橙
C_INTERACT = (252, 231, 243)  # 浅粉
C_DIAMOND  = (254, 215, 170)  # 橙(判断)
C_SIDE     = (243, 232, 255)  # 浅紫(旁支按键)
C_ARROW    = (75, 85, 99)
C_TITLE    = (15, 23, 42)

NODE_H = 58

# ========= 泳道 ============
LANES = [
    ("采集层", 0,       650, C_COLLECT),
    ("识别层", 650,     1420, C_RECOG),
    ("控制层", 1420,    2090, C_CTRL),
    ("交互层", 2090,    2790, C_INTERACT),
]
MARGIN_L = 50
MARGIN_R = 2800 - 50

LANE_HEADER_COLORS = [
    (191, 219, 254),
    (187, 247, 208),
    (253, 230, 138),
    (251, 207, 232),
]

# ========= 节点定义 ============
# (x_c, y_c, w, h, text, fill, kind=rect/diamond/round/doc/circle)
NODES = []

def add_rect(xc, yc, w, h, text, fill):
    NODES.append(("rect", xc - w // 2, yc - h // 2, xc + w // 2, yc + h // 2, text, fill))

def add_diamond(xc, yc, w, h, text, fill):
    NODES.append(("diamond", xc, yc, w, h, text, fill))

def add_doc(xc, yc, w, h, text, fill):
    NODES.append(("doc", xc - w // 2, yc - h // 2, xc + w // 2, yc + h // 2, text, fill))

# 采集层(0-650)
add_rect(325, 180, 500, NODE_H, "① 启动程序\ncv2.VideoCapture(0)", C_COLLECT)
add_rect(325, 300, 500, NODE_H, "② 摄像头取帧\n640×480 @ 30FPS", C_COLLECT)
add_rect(325, 420, 500, NODE_H, "③ 水平镜像翻转 + 亮度增强\n(对比度×1.2 亮度+35 Gamma 0.85)", C_COLLECT)
add_rect(325, 560, 520, NODE_H + 4, "④ MediaPipe Hands 推理\n提取 21 个手部 3D 关键点", C_COLLECT)

# 识别层(650-1420)
add_diamond(975, 180, 220, 130, "检测到\n手部?", C_DIAMOND)
add_rect(975, 340, 540, NODE_H, "⑤ 关键点归一化\n(手腕为原点 除以最大半径)", C_RECOG)
add_rect(975, 460, 540, NODE_H, "⑥ 特征提取：\ncount_fingers 数指 + 向量夹角 + 指距比", C_RECOG)
add_rect(975, 600, 620, NODE_H + 6, "⑦ recognize 优先级调度\n六字>剪刀>OK>食指>拇指>五指>两指>握拳", C_RECOG)
add_rect(975, 740, 540, NODE_H, "⑧ 自定义手势识别\n归一化关键点 平均绝对距离 阈值匹配", C_RECOG)

# 控制层(1420-2090)
X_CTRL = 1755
add_rect(X_CTRL, 180, 620, NODE_H, "⑨ 防抖状态机\n(同手势持续1秒触发 + 1秒冷却期)", C_CTRL)
add_diamond(X_CTRL, 320, 230, 140, "是否\n触发?", C_DIAMOND)
add_rect(X_CTRL - 180, 470, 330, NODE_H, "⑩ 仅更新 HUD 防抖进度条\n0% ~ 99% 黄渐变", C_CTRL)
add_doc(X_CTRL + 190, 470, 360, NODE_H + 4, "⑪ PyAutoGUI 模拟键盘按键\nright / left / home /\npageup / pagedown / esc", C_CTRL)

# 交互层(2090-2790)
X_INT = 2440
add_doc(X_INT, 300, 480, NODE_H + 4, "⑫ 中心绿色弹窗反馈\n(已执行:下一页/上一页 等 1.2s 淡出)", C_INTERACT)
add_rect(X_INT, 430, 500, NODE_H, "⑬ Pillow 中文渲染 HUD\n当前手势 FPS 亮度 全屏状态", C_INTERACT)
add_rect(X_INT, 560, 500, NODE_H, "⑭ 等比例缩放 + 居中黑边\n支持 F 键全屏切换", C_INTERACT)
add_doc(X_INT, 690, 460, NODE_H, "⑮ cv2.imshow 显示画面", C_INTERACT)
add_diamond(X_INT, 830, 240, 140, "Q / ESC 或\n退出手势?", C_DIAMOND)
add_doc(X_INT, 970, 520, NODE_H + 4, "⑯ 资源释放\ncap.release() + cv2.destroyAllWindows()", C_INTERACT)
add_rect(X_INT, 1100, 300, NODE_H, "结束 ✅", (224, 231, 255))

# 旁支按键: U V W X 连到⑬(HUD)
SIDES = [
    (45, 1260, "U  R 键",          "录制新手势\n保存为 .pkl + .json"),
    (45, 1360, "V  F 键",          "全屏 / 窗口 切换"),
    (45, 1460, "W  空格键",        "暂停 / 恢复检测"),
    (45, 1560, "X  [/] ,/.  0 键", "亮度 / 对比度 / Gamma 调节"),
]


# ========== 绘制 ==========
img = Image.new("RGB", (CW, CH), BG)
draw = ImageDraw.Draw(img)

def text_bbox(text, font):
    try:
        b = draw.textbbox((0, 0), text, font=font)
        return (b[2] - b[0], b[3] - b[1])
    except Exception:
        return (len(text) * 14, 22)

def draw_multiline_text_center(xc, yc, text, font, color=BLACK):
    lines = text.split("\n")
    total_h = sum(text_bbox(line, font)[1] + 4 for line in lines) - 4
    y = yc - total_h // 2
    for line in lines:
        tw, th = text_bbox(line, font)
        x = xc - tw // 2
        draw.text((x, y), line, font=font, fill=color)
        y += th + 4

def draw_rounded_rect(x1, y1, x2, y2, radius, fill, outline=DARK, width=2):
    draw.rounded_rectangle((x1, y1, x2, y2), radius=radius, fill=fill, outline=outline, width=width)

def draw_diamond(xc, yc, w, h, fill, outline=DARK, width=2):
    pts = [
        (xc, yc - h // 2),
        (xc + w // 2, yc),
        (xc, yc + h // 2),
        (xc - w // 2, yc),
    ]
    draw.polygon(pts, fill=fill, outline=outline)
    # 加粗
    for i in range(1, width):
        shrink = i
        pts2 = [
            (xc, yc - h // 2 + shrink),
            (xc + w // 2 - shrink, yc),
            (xc, yc + h // 2 - shrink),
            (xc - w // 2 + shrink, yc),
        ]
        draw.line(pts2 + [pts2[0]], fill=outline, width=1)

def draw_doc(x1, y1, x2, y2, fill, outline=DARK, width=2):
    # 平行四边形风格 (带文档右波)
    w, h = x2 - x1, y2 - y1
    skew = 20
    pts = [(x1 + skew, y1), (x2, y1), (x2 - skew, y2), (x1, y2)]
    draw.polygon(pts, fill=fill, outline=outline)
    # 右波线
    for _ in range(1, width):
        pass
    # 内部 3 条横线
    for i in range(3):
        yy = y1 + int(h * (0.30 + i * 0.18))
        draw.line((x1 + skew + 18, yy, x2 - skew - 18, yy), fill=DARK, width=1)

def arrow(x1, y1, x2, y2, color=C_ARROW, width=3, dashed=False, label=None, label_pos=None, label_bg=(255,255,255,220)):
    if dashed:
        steps = int(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 // 14)
        if steps < 2:
            steps = 2
        for i in range(steps):
            t1 = i / steps
            t2 = (i + 0.55) / steps
            px1 = int(x1 + (x2 - x1) * t1)
            py1 = int(y1 + (y2 - y1) * t1)
            px2 = int(x1 + (x2 - x1) * t2)
            py2 = int(y1 + (y2 - y1) * t2)
            draw.line((px1, py1, px2, py2), fill=color, width=width)
    else:
        draw.line((x1, y1, x2, y2), fill=color, width=width)
    # 箭头三角
    import math
    ang = math.atan2(y2 - y1, x2 - x1)
    ah = 14
    aw = 9
    p1 = (x2, y2)
    p2 = (x2 - ah * math.cos(ang) + aw * math.sin(ang),
          y2 - ah * math.sin(ang) - aw * math.cos(ang))
    p3 = (x2 - ah * math.cos(ang) - aw * math.sin(ang),
          y2 - ah * math.sin(ang) + aw * math.cos(ang))
    draw.polygon([p1, p2, p3], fill=color)
    if label:
        lx, ly = (x1 + x2) // 2, (y1 + y2) // 2
        if label_pos == "r":
            lx += 6
        elif label_pos == "l":
            lx -= 6
        tw, th = text_bbox(label, F14)
        draw.rounded_rectangle((lx - tw // 2 - 6, ly - th // 2 - 3, lx + tw // 2 + 6, ly + th // 2 + 3),
                               radius=6, fill=(255, 255, 255, 220), outline=GRAY, width=1)
        draw_multiline_text_center(lx, ly, label, F14, BLACK)


# ======= 绘制背景泳道 ========
draw.rounded_rectangle((MARGIN_L, 80, MARGIN_R, 1870), radius=18, fill=WHITE, outline=GRAY, width=2)

cur_x = MARGIN_L
for i, (name, x_start, x_end, _fill) in enumerate(LANES):
    x1 = cur_x if i == 0 else x_start
    x2 = x_end if i < len(LANES) - 1 else MARGIN_R
    hdr = LANE_HEADER_COLORS[i]
    draw.rounded_rectangle((x1, 80, x2, 130), radius=10, fill=hdr, outline=GRAY, width=1)
    tw, th = text_bbox(name, F22B)
    draw.text(((x1 + x2) // 2 - tw // 2, 105 - th // 2), name, font=F22B, fill=C_TITLE)
    # 泳道分隔竖线(除最后)
    if i < len(LANES) - 1:
        draw.line((x_end, 80, x_end, 1870), fill=GRAY, width=2)
    # 泳道底色半透明
    overlay = Image.new("RGBA", (CW, CH), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    od.rectangle((x1 + 1, 131, x2 - 1, 1869), fill=(_fill[0], _fill[1], _fill[2], 90))
    img.paste(Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB"))
    # 恢复 draw
    draw = ImageDraw.Draw(img)

# 标题
TITLE = "图 2-1  手势识别 PPT 控制系统 总体流程图"
tw, th = text_bbox(TITLE, F28B)
draw.text((CW // 2 - tw // 2, 22), TITLE, font=F28B, fill=C_TITLE)

SUB = " (4 层泳道：采集层 → 识别层 → 控制层 → 交互层，共 16 个处理节点 + 4 个旁支按键交互)"
tw2, _ = text_bbox(SUB, F16)
draw.text((CW // 2 - tw2 // 2, 60), SUB, font=F16, fill=DARK)

# ========= 绘制节点 =========
node_rect_by_name = {}  # 存中心坐标
node_centers = []
for nd in NODES:
    kind = nd[0]
    if kind == "rect":
        _, x1, y1, x2, y2, text, fill = nd
        draw_rounded_rect(x1, y1, x2, y2, 12, fill)
        xc, yc = (x1 + x2) // 2, (y1 + y2) // 2
        draw_multiline_text_center(xc, yc, text, F16, BLACK)
        node_centers.append((xc, yc, text, fill))
    elif kind == "diamond":
        _, xc, yc, w, h, text, fill = nd
        draw_diamond(xc, yc, w, h, fill)
        draw_multiline_text_center(xc, yc, text, F16, BLACK)
        node_centers.append((xc, yc, text, fill))
    elif kind == "doc":
        _, x1, y1, x2, y2, text, fill = nd
        draw_doc(x1, y1, x2, y2, fill)
        xc, yc = (x1 + x2) // 2, (y1 + y2) // 2
        draw_multiline_text_center(xc, yc, text, F16, BLACK)
        node_centers.append((xc, yc, text, fill))

# ========= 绘制旁支按键 =========
side_outs = []
for (sx, sy, title, desc) in SIDES:
    # 紫色圆角块 (R/F/空格/调节键)
    bx1, by1 = sx, sy
    bx2, by2 = bx1 + 380, by1 + 72
    draw_rounded_rect(bx1, by1, bx2, by2, 12, C_SIDE)
    # 标题
    draw.text((bx1 + 14, by1 + 8), title, font=F18, fill=(88, 28, 135))
    # 描述
    draw_multiline_text_center((bx1 + bx2) // 2, by1 + 52, desc, F14, BLACK)
    # 箭头右端指向 HUD 节点(第18个 X_INT,430: Pillow中文渲染HUD)
    HUD_X, HUD_Y = 2440, 430
    side_outs.append(((bx2, (by1 + by2) // 2), (HUD_X - 250, HUD_Y)))

# 旁支箭头: 先水平再转折
for (start, end_) in side_outs:
    ex, ey = end_
    sx, sy = start
    mid_x = 1500
    draw.line((sx, sy, mid_x, sy), fill=(139, 92, 246), width=3)
    draw.line((mid_x, sy, mid_x, ey), fill=(139, 92, 246), width=3)
    arrow(mid_x, ey, ex, ey, color=(139, 92, 246), width=3)

# 绘制主流程箭头(按照我们想要的连接关系)
# 节点索引与 node_centers 对应: 按 NODES 插入顺序
# 0:启动 1:取帧 2:翻转+亮度 3:MP推理 4:判断手? 5:归一化 6:特征 7:优先级 8:自定义
# 9:防抖 10:是否触发? 11:进度条 12:PyAutoGUI
# 13:弹窗 14:HUD 15:缩放 16:imshow 17:退出? 18:释放 19:结束

def cx(i): return node_centers[i][0]
def cy(i): return node_centers[i][1]
def bottom(i): return node_centers[i][1] + (NODE_H // 2 if i not in (4, 10, 17) else 65)
def top(i): return node_centers[i][1] - (NODE_H // 2 if i not in (4, 10, 17) else 65)
def right(i): return node_centers[i][0] + 250
def left(i): return node_centers[i][0] - 250

# 0→1→2→3 (竖直 同列)
arrow(cx(0), bottom(0), cx(1), top(1))
arrow(cx(1), bottom(1), cx(2), top(2))
arrow(cx(2), bottom(2), cx(3), top(3))
# 3→4 (跨泳道)
arrow(cx(3) + 260, cy(3), cx(4) - 110, cy(4))
# 4:判断手? - 否→ (向右到 进度条11, 其实应该是箭头指向HUD直接画, 我们画到 "空手直接到HUD": 从 4 向右下到 14(HUD))
# 4 右出口(否)→14 HUD
arrow(cx(4) + 110, cy(4), cx(14) - 250, cy(14) + 20, label="否(空手)", label_bg=(255,255,240,220))
# 4 下出口(是)→5
arrow(cx(4), bottom(4), cx(5), top(5), label="是", label_pos="r")
# 5→6→7→8 (同列)
arrow(cx(5), bottom(5), cx(6), top(6))
arrow(cx(6), bottom(6), cx(7), top(7))
arrow(cx(7), bottom(7), cx(8), top(8))
# 8→9 (跨泳道)
arrow(cx(8) + 270, cy(8), cx(9) - 310, cy(9))
# 9→10
arrow(cx(9), bottom(9), cx(10), top(10))
# 10 否 → 11(进度条)
arrow(cx(10) - 115, cy(10), cx(11) + 165, cy(11), label="否", label_pos="r")
# 10 是 → 12(PyAutoGUI)
arrow(cx(10) + 115, cy(10), cx(12) - 180, cy(12), label="是", label_pos="r")
# 11→14 HUD, 12→13 弹窗
arrow(cx(11) + 165, cy(11), cx(14) - 250, cy(14))
arrow(cx(12) + 180, cy(12), cx(13) - 240, cy(13))
# 13→14 HUD
arrow(cx(13), bottom(13) + 4, cx(14), top(14) - 4)
# 14→15→16→17
arrow(cx(14), bottom(14), cx(15), top(15))
arrow(cx(15), bottom(15), cx(16), top(16))
arrow(cx(16), bottom(16), cx(17), top(17))
# 17 否 → 回到 取帧节点(1) top
# 画折线: 从 17 右侧向右, 再向上, 再向左回到 取帧左
src_x = cx(17) + 120
src_y = cy(17)
mid_x = 2740
mid_y_top = 300
draw.line((src_x, src_y, mid_x, src_y), fill=C_ARROW, width=3)
draw.line((mid_x, src_y, mid_x, mid_y_top), fill=C_ARROW, width=3)
back_mid_x = 325 + 250
draw.line((mid_x, mid_y_top, back_mid_x, mid_y_top), fill=C_ARROW, width=3, )
# 最后指向取帧节点顶部? 其实应该回到 ② 取帧(重复循环), 但取帧在第1列. 为避免与 1→2 冲突, 画到 ② 的右侧, 再折到 top
reframe_y = cy(1) - 29
draw.line((back_mid_x, mid_y_top, cx(1) + 120, mid_y_top), fill=C_ARROW, width=3)
arrow(cx(1) + 120, mid_y_top, cx(1) + 120, top(1) + 4, color=C_ARROW, width=3)
# 加文字"否 继续循环"
draw.rounded_rectangle((mid_x - 145, mid_y_top - 28, mid_x - 10, mid_y_top - 2), radius=6,
                       fill=(255, 255, 255, 230), outline=GRAY, width=1)
draw.text((mid_x - 138, mid_y_top - 22), "否 → 继续循环下一帧", font=F14, fill=BLACK)

# 17 是 → 18 释放
arrow(cx(17), bottom(17), cx(18), top(18), label="是", label_pos="r")
# 18→19 结束
arrow(cx(18), bottom(18), cx(19), top(19))


# ======== 右下角图例 ============
LEG_X, LEG_Y = 2300, 1730
draw_rounded_rect(LEG_X, LEG_Y, LEG_X + 420, LEG_Y + 120, 10, (255, 255, 255), outline=GRAY)
draw.text((LEG_X + 18, LEG_Y + 8), "图例", font=F18, fill=C_TITLE)
# 块 1
dx1, dy1 = LEG_X + 22, LEG_Y + 38
draw_rounded_rect(dx1, dy1, dx1 + 26, dy1 + 18, 5, C_COLLECT, outline=DARK)
draw.text((dx1 + 34, dy1), "处理节点", font=F14, fill=BLACK)
dx1 += 150
draw_diamond(dx1 + 13, dy1 + 9, 34, 22, C_DIAMOND, outline=DARK)
draw.text((dx1 + 34, dy1), "判断节点", font=F14, fill=BLACK)
dx2, dy2 = LEG_X + 22, LEG_Y + 68
draw_doc(dx2, dy2 - 2, dx2 + 26, dy2 + 20, (254, 215, 170), outline=DARK)
draw.text((dx2 + 34, dy2), "I/O 文档/按键", font=F14, fill=BLACK)
dx2 += 170
draw_rounded_rect(dx2, dy2 - 2, dx2 + 26, dy2 + 20, 5, C_SIDE, outline=DARK)
draw.text((dx2 + 34, dy2), "旁支按键交互", font=F14, fill=BLACK)

# 底部来源
src = "生成脚本:draw_flowchart.py     对应报告:项目报告_手势识别PPT控制.md 第2.1节"
tw, th = text_bbox(src, F14)
draw.text((CW // 2 - tw // 2, 1885), src, font=F14, fill=DARK)

img.save(OUT_FILE, "PNG", dpi=(150, 150))
print("已生成:", OUT_FILE, "尺寸:", CW, "x", CH)
