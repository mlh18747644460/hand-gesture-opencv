import cv2
import numpy as np
import time
import os
import json
import pickle
from collections import deque
from typing import Dict, List, Tuple, Optional, Any

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("错误：缺少 Pillow 库，请运行: pip install pillow")
    raise

try:
    import mediapipe as mp
except ImportError:
    print("错误：缺少 mediapipe 库，请运行: pip install mediapipe")
    raise

try:
    import pyautogui
except ImportError:
    print("错误：缺少 pyautogui 库，请运行: pip install pyautogui")
    raise


GESTURE_PALM = "palm"
GESTURE_TWO_FINGERS = "two_fingers"
GESTURE_THUMB_UP = "thumb_up"
GESTURE_OK = "ok"
GESTURE_INDEX_UP = "index_up"
GESTURE_SCISSORS = "scissors"
GESTURE_SIX = "six"
GESTURE_FIST = "fist"
GESTURE_NONE = "none"
GESTURE_UNKNOWN = "unknown"

ACTION_NEXT_PAGE = "next_page"
ACTION_PREV_PAGE = "prev_page"
ACTION_EXIT = "exit"
ACTION_HOME = "home"
ACTION_SCROLL_UP = "scroll_up"
ACTION_SCROLL_DOWN = "scroll_down"
ACTION_BACK_PAGE = "back_page"
ACTION_CUSTOM = "custom"

DEFAULT_GESTURE_MAP = {
    GESTURE_PALM: ACTION_PREV_PAGE,
    GESTURE_FIST: ACTION_NEXT_PAGE,
    GESTURE_OK: ACTION_HOME,
    GESTURE_INDEX_UP: ACTION_SCROLL_UP,
    GESTURE_SCISSORS: ACTION_SCROLL_DOWN,
    GESTURE_SIX: ACTION_BACK_PAGE,
}

ACTION_TO_KEY = {
    ACTION_NEXT_PAGE: "right",
    ACTION_PREV_PAGE: "left",
    ACTION_BACK_PAGE: "left",
    ACTION_EXIT: "esc",
    ACTION_HOME: "home",
    ACTION_SCROLL_UP: "pageup",
    ACTION_SCROLL_DOWN: "pagedown",
}

GESTURE_NAMES_CN = {
    GESTURE_PALM: "张开手掌 上一页",
    GESTURE_FIST: "握拳 下一页",
    GESTURE_OK: "OK手势 从头播放",
    GESTURE_INDEX_UP: "食指单独上指 向上滚动",
    GESTURE_SCISSORS: "剪刀手势 向下滚动",
    GESTURE_SIX: "六字手势 返回上一页",
    GESTURE_TWO_FINGERS: "两指伸出 请对准镜头",
    GESTURE_THUMB_UP: "竖拇指 请对准镜头",
    GESTURE_NONE: "空手 未检测到手",
    GESTURE_UNKNOWN: "请对准镜头 做清晰手势",
}

ACTION_HINT_CN = {
    ACTION_NEXT_PAGE: "下一页",
    ACTION_PREV_PAGE: "上一页",
    ACTION_BACK_PAGE: "返回上一页",
    ACTION_EXIT: "退出程序",
    ACTION_HOME: "从头播放",
    ACTION_SCROLL_UP: "向上滚动",
    ACTION_SCROLL_DOWN: "向下滚动",
    ACTION_CUSTOM: "自定义按键",
}

CUSTOM_GESTURES_FILE = "custom_gestures.pkl"
CUSTOM_BINDINGS_FILE = "custom_bindings.json"


class HandGestureRecognizer:
    def __init__(self, static_image_mode: bool = False,
                 max_num_hands: int = 1,
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

        self.tip_ids = [4, 8, 12, 16, 20]
        self.pip_ids = [3, 6, 10, 14, 18]

        self.custom_gestures: Dict[str, List[List[float]]] = {}
        self._load_custom_gestures()

    def _load_custom_gestures(self):
        if os.path.exists(CUSTOM_GESTURES_FILE):
            try:
                with open(CUSTOM_GESTURES_FILE, 'rb') as f:
                    self.custom_gestures = pickle.load(f)
            except Exception:
                self.custom_gestures = {}

    def _save_custom_gestures(self):
        with open(CUSTOM_GESTURES_FILE, 'wb') as f:
            pickle.dump(self.custom_gestures, f)

    def record_custom_gesture(self, name: str, landmarks_list: List[List[float]]):
        self.custom_gestures[name] = landmarks_list
        self._save_custom_gestures()

    def list_custom_gestures(self) -> List[str]:
        return list(self.custom_gestures.keys())

    def delete_custom_gesture(self, name: str):
        if name in self.custom_gestures:
            del self.custom_gestures[name]
            self._save_custom_gestures()

    def process_frame(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.hands.process(rgb)

    def draw_landmarks(self, frame: np.ndarray, hand_landmarks) -> np.ndarray:
        if hand_landmarks:
            self.mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_styles.get_default_hand_landmarks_style(),
                self.mp_styles.get_default_hand_connections_style()
            )
        return frame

    def extract_landmarks(self, hand_landmarks, img_shape: Tuple[int, int]) -> Optional[List[float]]:
        if not hand_landmarks:
            return None
        h, w = img_shape[:2]
        lm = []
        for landmark in hand_landmarks.landmark:
            lm.append(landmark.x)
            lm.append(landmark.y)
            lm.append(landmark.z)
        return lm

    def _normalize_landmarks(self, lm: List[float]) -> List[float]:
        if len(lm) < 3:
            return lm
        xs = lm[0::3]
        ys = lm[1::3]
        zs = lm[2::3]

        wrist_x, wrist_y, wrist_z = xs[0], ys[0], zs[0]

        max_dist = 0.0
        for i in range(len(xs)):
            dx = xs[i] - wrist_x
            dy = ys[i] - wrist_y
            dist = (dx ** 2 + dy ** 2) ** 0.5
            if dist > max_dist:
                max_dist = dist

        if max_dist < 1e-6:
            max_dist = 1.0

        normalized = []
        for i in range(len(xs)):
            normalized.append((xs[i] - wrist_x) / max_dist)
            normalized.append((ys[i] - wrist_y) / max_dist)
            normalized.append((zs[i] - wrist_z) / max_dist)
        return normalized

    def count_fingers(self, hand_landmarks, handedness_label: str) -> int:
        if not hand_landmarks:
            return 0
        fingers = []
        landmarks = hand_landmarks.landmark

        if handedness_label == 'Right':
            if landmarks[self.tip_ids[0]].x < landmarks[self.tip_ids[0] - 1].x:
                fingers.append(1)
            else:
                fingers.append(0)
        else:
            if landmarks[self.tip_ids[0]].x > landmarks[self.tip_ids[0] - 1].x:
                fingers.append(1)
            else:
                fingers.append(0)

        for i in range(1, 5):
            tip_y = landmarks[self.tip_ids[i]].y
            pip_y = landmarks[self.pip_ids[i]].y
            if tip_y < pip_y:
                fingers.append(1)
            else:
                fingers.append(0)

        return sum(fingers)

    def _vector_2d(self, landmarks, i1: int, i2: int) -> Tuple[float, float]:
        p1 = landmarks[i1]
        p2 = landmarks[i2]
        return (p2.x - p1.x, p2.y - p1.y)

    def _vector_angle_deg(self, v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        n1 = (v1[0] ** 2 + v1[1] ** 2) ** 0.5
        n2 = (v2[0] ** 2 + v2[1] ** 2) ** 0.5
        if n1 < 1e-9 or n2 < 1e-9:
            return 180.0
        cosv = max(-1.0, min(1.0, dot / (n1 * n2)))
        return np.degrees(np.arccos(cosv))

    def _is_thumb_up(self, hand_landmarks) -> bool:
        if not hand_landmarks:
            return False
        lm = hand_landmarks.landmark

        thumb_tip_y = lm[self.tip_ids[0]].y
        thumb_ip_y = lm[3].y
        thumb_mcp_y = lm[2].y
        wrist_y = lm[0].y

        thumb_pointing_up = (thumb_tip_y < thumb_ip_y < thumb_mcp_y < wrist_y)

        four_fingers_folded = True
        for i in range(1, 5):
            tip = lm[self.tip_ids[i]]
            pip = lm[self.pip_ids[i]]
            mcp = lm[self.pip_ids[i] - 1] if self.pip_ids[i] > 0 else lm[0]
            if tip.y < pip.y:
                four_fingers_folded = False
                break

        tip_x = lm[self.tip_ids[0]].x
        mcp_x = lm[2].x
        angle_ok = True
        if not (abs(tip_x - mcp_x) < 0.15):
            angle_ok = False

        v1 = self._vector_2d(lm, 2, 3)
        v2 = self._vector_2d(lm, 3, 4)
        angle = self._vector_angle_deg(v1, v2)
        straight = angle < 40.0

        return thumb_pointing_up and four_fingers_folded and straight

    def _is_ok_gesture(self, hand_landmarks) -> bool:
        if not hand_landmarks:
            return False
        lm = hand_landmarks.landmark

        thumb_tip = lm[4]
        index_tip = lm[8]
        dx = thumb_tip.x - index_tip.x
        dy = thumb_tip.y - index_tip.y
        dist_touch = (dx ** 2 + dy ** 2) ** 0.5

        thumb_ip = lm[3]
        index_pip = lm[6]
        dx2 = thumb_ip.x - index_pip.x
        dy2 = thumb_ip.y - index_pip.y
        ref_dist = max((dx2 ** 2 + dy2 ** 2) ** 0.5, 0.02)

        touching = (dist_touch / ref_dist) < 1.2

        three_fingers_extended = True
        for i in range(2, 5):
            tip_y = lm[self.tip_ids[i]].y
            pip_y = lm[self.pip_ids[i]].y
            if tip_y >= pip_y:
                three_fingers_extended = False
                break

        return touching and three_fingers_extended

    def _is_index_up(self, hand_landmarks, handedness_label: str) -> bool:
        fingers = self.count_fingers(hand_landmarks, handedness_label)
        if fingers != 1:
            return False
        if not hand_landmarks:
            return False
        lm = hand_landmarks.landmark
        index_tip_y = lm[8].y
        index_pip_y = lm[6].y
        middle_tip_y = lm[12].y
        middle_pip_y = lm[10].y
        return index_tip_y < index_pip_y and middle_tip_y > middle_pip_y

    def _is_scissors_gesture(self, hand_landmarks, handedness_label: str) -> bool:
        if not hand_landmarks:
            return False
        fingers = self.count_fingers(hand_landmarks, handedness_label)
        if fingers != 2:
            return False
        lm = hand_landmarks.landmark
        index_extended = lm[8].y < lm[6].y
        middle_extended = lm[12].y < lm[10].y
        ring_folded = lm[16].y > lm[14].y
        pinky_folded = lm[20].y > lm[18].y
        if not (index_extended and middle_extended and ring_folded and pinky_folded):
            return False

        idx_v = self._vector_2d(lm, 5, 8)
        mid_v = self._vector_2d(lm, 9, 12)
        angle = self._vector_angle_deg(idx_v, mid_v)
        separated = (10.0 < angle < 70.0)
        tip_dx = lm[8].x - lm[12].x
        tip_dy = lm[8].y - lm[12].y
        tip_d = (tip_dx ** 2 + tip_dy ** 2) ** 0.5
        ref_v = self._vector_2d(lm, 0, 9)
        ref_d = max((ref_v[0] ** 2 + ref_v[1] ** 2) ** 0.5, 0.02)
        far_enough = (tip_d / ref_d) > 0.12
        return separated or far_enough

    def _is_six_gesture(self, hand_landmarks, handedness_label: str) -> bool:
        if not hand_landmarks:
            return False
        fingers = self.count_fingers(hand_landmarks, handedness_label)
        if fingers != 2:
            return False
        lm = hand_landmarks.landmark

        thumb_extended = False
        if handedness_label == 'Right':
            thumb_extended = (lm[4].x < lm[3].x < lm[2].x)
        else:
            thumb_extended = (lm[4].x > lm[3].x > lm[2].x)
        if not thumb_extended:
            return False

        pinky_extended = lm[20].y < lm[18].y
        if not pinky_extended:
            return False

        index_folded = lm[8].y > lm[6].y
        middle_folded = lm[12].y > lm[10].y
        ring_folded = lm[16].y > lm[14].y
        if not (index_folded and middle_folded and ring_folded):
            return False

        thumb_tip = lm[4]
        pinky_tip = lm[20]
        idx_tip = lm[8]
        d_tp_x = thumb_tip.x - pinky_tip.x
        d_tp_y = thumb_tip.y - pinky_tip.y
        d_tp = (d_tp_x ** 2 + d_tp_y ** 2) ** 0.5
        d_ti_x = thumb_tip.x - idx_tip.x
        d_ti_y = thumb_tip.y - idx_tip.y
        d_ti = max((d_ti_x ** 2 + d_ti_y ** 2) ** 0.5, 0.02)
        spread_ok = (d_tp / d_ti) > 0.7
        return spread_ok

    def _match_custom_gesture(self, lm: List[float], threshold: float = 0.12) -> Optional[str]:
        if not self.custom_gestures or not lm:
            return None

        norm_lm = self._normalize_landmarks(lm)

        best_name = None
        best_dist = float('inf')
        for name, samples in self.custom_gestures.items():
            min_d = float('inf')
            for sample in samples:
                if len(sample) != len(norm_lm):
                    continue
                norm_sample = self._normalize_landmarks(sample)
                arr1 = np.array(norm_lm)
                arr2 = np.array(norm_sample)
                d = float(np.mean(np.abs(arr1 - arr2)))
                if d < min_d:
                    min_d = d
            if min_d < best_dist:
                best_dist = min_d
                best_name = name

        if best_dist < threshold:
            return best_name
        return None

    def recognize(self, hand_landmarks, handedness_label: str, raw_lm: Optional[List[float]] = None) -> str:
        if hand_landmarks is None:
            return GESTURE_NONE

        custom = None
        if raw_lm is not None:
            custom = self._match_custom_gesture(raw_lm)
        if custom:
            return custom

        fingers = self.count_fingers(hand_landmarks, handedness_label)

        if self._is_six_gesture(hand_landmarks, handedness_label):
            return GESTURE_SIX

        if self._is_scissors_gesture(hand_landmarks, handedness_label):
            return GESTURE_SCISSORS

        if self._is_ok_gesture(hand_landmarks):
            return GESTURE_OK

        if self._is_index_up(hand_landmarks, handedness_label):
            return GESTURE_INDEX_UP

        if self._is_thumb_up(hand_landmarks):
            return GESTURE_THUMB_UP

        if fingers == 5:
            return GESTURE_PALM

        if fingers == 2:
            return GESTURE_TWO_FINGERS

        if fingers == 0:
            return GESTURE_FIST

        return GESTURE_UNKNOWN


class DebounceStateMachine:
    def __init__(self, debounce_time: float = 1.0, cooldown_time: float = 1.0):
        self.debounce_time = debounce_time
        self.cooldown_time = cooldown_time
        self.current_gesture = GESTURE_NONE
        self.gesture_start_time = 0.0
        self.last_triggered_gesture = GESTURE_NONE
        self.last_trigger_time = 0.0
        self.triggered_in_streak = False
        self.history = deque(maxlen=10)

    def update(self, detected_gesture: str, current_time: Optional[float] = None) -> Tuple[str, bool, float]:
        if current_time is None:
            current_time = time.time()

        self.history.append(detected_gesture)

        if detected_gesture != self.current_gesture:
            self.current_gesture = detected_gesture
            self.gesture_start_time = current_time
            self.triggered_in_streak = False

        elapsed = current_time - self.gesture_start_time
        stable = (elapsed >= self.debounce_time)

        in_cooldown = (current_time - self.last_trigger_time) < self.cooldown_time

        should_trigger = False
        if stable and not self.triggered_in_streak and not in_cooldown:
            if self.current_gesture != GESTURE_NONE and self.current_gesture != GESTURE_UNKNOWN:
                should_trigger = True
                self.triggered_in_streak = True
                self.last_triggered_gesture = self.current_gesture
                self.last_trigger_time = current_time

        return self.current_gesture, should_trigger, elapsed

    def get_progress(self) -> float:
        if self.current_gesture in (GESTURE_NONE, GESTURE_UNKNOWN):
            return 0.0
        elapsed = time.time() - self.gesture_start_time
        return min(1.0, elapsed / self.debounce_time)

    def reset(self):
        self.current_gesture = GESTURE_NONE
        self.gesture_start_time = 0.0
        self.last_triggered_gesture = GESTURE_NONE
        self.last_trigger_time = 0.0
        self.triggered_in_streak = False
        self.history.clear()


class KeyboardController:
    def __init__(self, fail_safe: bool = True):
        self.fail_safe = fail_safe
        if fail_safe:
            pyautogui.FAILSAFE = True

    def execute_action(self, action: str, custom_key: Optional[str] = None) -> bool:
        try:
            if action == ACTION_CUSTOM and custom_key:
                keys = custom_key.lower().split('+')
                if len(keys) > 1:
                    pyautogui.hotkey(*[k.strip() for k in keys])
                else:
                    pyautogui.press(keys[0].strip())
                return True

            key = ACTION_TO_KEY.get(action)
            if key:
                pyautogui.press(key)
                return True
            return False
        except Exception as e:
            print(f"键盘控制错误: {e}")
            return False


class CustomBindingManager:
    def __init__(self):
        self.bindings: Dict[str, str] = {}
        self._load()

    def _load(self):
        if os.path.exists(CUSTOM_BINDINGS_FILE):
            try:
                with open(CUSTOM_BINDINGS_FILE, 'r', encoding='utf-8') as f:
                    self.bindings = json.load(f)
            except Exception:
                self.bindings = {}

    def _save(self):
        with open(CUSTOM_BINDINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.bindings, f, ensure_ascii=False, indent=2)

    def bind(self, gesture_name: str, key_string: str):
        self.bindings[gesture_name] = key_string
        self._save()

    def unbind(self, gesture_name: str):
        if gesture_name in self.bindings:
            del self.bindings[gesture_name]
            self._save()

    def get_key(self, gesture_name: str) -> Optional[str]:
        return self.bindings.get(gesture_name)


class GesturePPTController:
    def __init__(self, camera_id: int = 0, debounce_time: float = 1.0,
                 target_fps: int = 30, show_fps: bool = True):
        self.camera_id = camera_id
        self.target_fps = target_fps
        self.show_fps = show_fps

        self.recognizer = HandGestureRecognizer(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.debounce = DebounceStateMachine(debounce_time=debounce_time, cooldown_time=1.0)
        self.keyboard = KeyboardController()
        self.bindings = CustomBindingManager()

        self.gesture_map = dict(DEFAULT_GESTURE_MAP)

        self.recording_mode = False
        self.recorded_samples: List[List[float]] = []
        self.record_target_name: Optional[str] = None
        self.record_min_samples = 10
        self.record_max_samples = 30
        self.record_last_frame_time = 0.0

        self.feedback_text: Optional[str] = None
        self.feedback_start_time = 0.0
        self.feedback_duration = 1.2

        self.brightness = 35
        self.contrast = 1.2
        self.gamma = 0.85
        self._gamma_table = self._build_gamma_table(self.gamma)

        self.font_sm, self.font_md, self.font_lg, self.font_xl = self._load_fonts()

        self.fullscreen = False
        self.screen_w, self.screen_h = self._detect_screen_size()
        self.WINDOW_TITLE = "Gesture PPT Control (手势控制, 按F全屏)"

        self.running = False

    @staticmethod
    def _detect_screen_size() -> Tuple[int, int]:
        try:
            sw = pyautogui.size()[0]
            sh = pyautogui.size()[1]
            return (int(sw), int(sh))
        except Exception:
            return (1920, 1080)

    def _apply_fullscreen(self, current_state: Optional[bool] = None):
        state = self.fullscreen if current_state is None else current_state
        try:
            if state:
                cv2.setWindowProperty(self.WINDOW_TITLE,
                                      cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(self.WINDOW_TITLE,
                                      cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_NORMAL)
        except Exception:
            pass

    def _toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        self._apply_fullscreen()
        self._set_feedback("全屏模式 已开启" if self.fullscreen else "已切换回 窗口模式")

    def _fit_to_screen(self, frame: np.ndarray) -> np.ndarray:
        if not self.fullscreen:
            return frame
        fh, fw = frame.shape[:2]
        sw, sh = self.screen_w, self.screen_h
        scale = min(sw / fw, sh / fh)
        nw = max(1, int(fw * scale))
        nh = max(1, int(fh * scale))
        if (nw, nh) != (fw, fh):
            frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        if nw == sw and nh == sh:
            return frame
        canvas = np.zeros((sh, sw, 3), dtype=np.uint8)
        ox = (sw - nw) // 2
        oy = (sh - nh) // 2
        canvas[oy:oy + nh, ox:ox + nw] = frame
        return canvas

    @staticmethod
    def _find_font_file() -> Optional[str]:
        candidates = [
            r"C:\Windows\Fonts\msyh.ttc",
            r"C:\Windows\Fonts\msyhbd.ttc",
            r"C:\Windows\Fonts\msyhl.ttc",
            r"C:\Windows\Fonts\simhei.ttf",
            r"C:\Windows\Fonts\simsun.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/truetype/arphic/ukai.ttc",
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Medium.ttc",
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        return None

    def _load_fonts(self) -> Tuple:
        path = self._find_font_file()
        if path is None:
            default = ImageFont.load_default()
            return default, default, default, default
        try:
            return (
                ImageFont.truetype(path, 15),
                ImageFont.truetype(path, 20),
                ImageFont.truetype(path, 28),
                ImageFont.truetype(path, 36),
            )
        except Exception:
            default = ImageFont.load_default()
            return default, default, default, default

    @staticmethod
    def _cv2_to_pil(frame: np.ndarray) -> "Image.Image":
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    @staticmethod
    def _pil_to_cv2(img: "Image.Image") -> np.ndarray:
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    def _pil_text_size(self, draw: "ImageDraw.ImageDraw", text: str,
                       font: "ImageFont.ImageFont") -> Tuple[int, int]:
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            return (bbox[2] - bbox[0], bbox[3] - bbox[1])
        except Exception:
            try:
                return draw.textsize(text, font=font)
            except Exception:
                return (len(text) * 16, 20)

    def _pil_text(self, draw: "ImageDraw.ImageDraw", pos: Tuple[int, int],
                  text: str, font: "ImageFont.ImageFont",
                  color: Tuple[int, int, int], stroke: int = 0,
                  stroke_color: Tuple[int, int, int] = (0, 0, 0)):
        draw.text(pos, text, font=font, fill=color,
                  stroke_width=stroke, stroke_fill=stroke_color)

    def _build_gamma_table(self, gamma: float) -> np.ndarray:
        inv_gamma = 1.0 / max(0.1, min(5.0, gamma))
        table = (np.arange(256) / 255.0) ** inv_gamma * 255.0
        return np.clip(table, 0, 255).astype(np.uint8)

    def _adjust_brightness(self, frame: np.ndarray) -> np.ndarray:
        if self.contrast != 1.0 or self.brightness != 0:
            frame = cv2.convertScaleAbs(frame, alpha=float(self.contrast), beta=float(self.brightness))
        if abs(self.gamma - 1.0) > 1e-3:
            frame = cv2.LUT(frame, self._gamma_table)
        return frame

    def _set_feedback(self, text: str):
        self.feedback_text = text
        self.feedback_start_time = time.time()

    def _draw_hud(self, frame: np.ndarray, gesture: str, progress: float,
                  fps: float, hand_exists: bool, hand_label: str = "") -> np.ndarray:
        h, w = frame.shape[:2]
        overlay = frame.copy()

        cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        pil_img = self._cv2_to_pil(frame)
        draw = ImageDraw.Draw(pil_img)

        if gesture not in (GESTURE_NONE, GESTURE_UNKNOWN):
            g_color = (0, 255, 0)
        elif gesture == GESTURE_NONE:
            g_color = (255, 255, 255)
        else:
            g_color = (255, 200, 0)
        gesture_cn = GESTURE_NAMES_CN.get(gesture, gesture)
        self._pil_text(draw, (15, 10), f"当前手势：{gesture_cn}",
                       self.font_lg, g_color, stroke=1)

        bar_w = 360
        bar_x = 15
        bar_y = 58
        bar_h = 18
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (70, 70, 70), -1)
        fill_w = int(bar_w * progress)
        bar_color = (0, 255, 0) if progress >= 1.0 else (0, 200, 255)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)
        frame = self._pil_to_cv2(pil_img)
        pil_img = self._cv2_to_pil(frame)
        draw = ImageDraw.Draw(pil_img)

        prog_text = f"防抖进度 {int(progress * 100):3d}%  (保持手势1秒触发)"
        self._pil_text(draw, (bar_x + 8, bar_y + 2), prog_text,
                       self.font_sm, (255, 255, 255))

        if hand_exists:
            hand_text = f"手部检测：已检测到  [{hand_label}]"
            h_color = (180, 230, 255)
        else:
            hand_text = "手部检测：空手状态  （无手势不操作）"
            h_color = (200, 200, 200)
        self._pil_text(draw, (15, 88), hand_text, self.font_md, h_color)

        if self.show_fps:
            fps_text = f"FPS: {fps:.0f}"
            tw, _ = self._pil_text_size(draw, fps_text, self.font_lg)
            self._pil_text(draw, (w - tw - 15, 10), fps_text,
                           self.font_lg, (0, 255, 255), stroke=1)

        fs_text = "全屏(按F切换)" if self.fullscreen else "窗口模式(F全屏)"
        fs_color = (80, 255, 120) if self.fullscreen else (220, 220, 255)
        ftw, _ = self._pil_text_size(draw, fs_text, self.font_sm)
        fs_x = max(15, w - ftw - 15)
        fs_y = 105
        self._pil_text(draw, (fs_x, fs_y), fs_text, self.font_sm, fs_color)

        info_line1 = f"亮度 {int(self.brightness):+3d}   对比 {self.contrast:.1f}倍   Gamma {self.gamma:.2f}"
        info_line2 = "按 [ 减亮 / ] 加亮    按 , 减对比 / . 加对比    按 0 重置所有"
        tw1, _ = self._pil_text_size(draw, info_line1, self.font_sm)
        tw2, _ = self._pil_text_size(draw, info_line2, self.font_sm)
        tw = max(tw1, tw2, ftw + 10)
        self._pil_text(draw, (w - tw - 15, 58), info_line1,
                       self.font_sm, (180, 255, 200))
        self._pil_text(draw, (w - tw - 15, 78), info_line2,
                       self.font_sm, (180, 230, 200))

        if self.recording_mode:
            cv2.rectangle(frame, (0, h - 90), (w, h), (0, 0, 180), -1)
            frame = self._pil_to_cv2(pil_img)
            pil_img = self._cv2_to_pil(frame)
            draw = ImageDraw.Draw(pil_img)
            line1 = f"录制模式：手势名『{self.record_target_name}』  样本数 {len(self.recorded_samples)}/{self.record_min_samples}个以上"
            line2 = "保持手势稳定  按 R 保存并退出    按 ESC 取消录制"
            self._pil_text(draw, (15, h - 80), line1, self.font_md, (255, 255, 255))
            self._pil_text(draw, (15, h - 45), line2, self.font_sm, (255, 255, 255))

        if self.feedback_text and (time.time() - self.feedback_start_time) < self.feedback_duration:
            alpha = 1.0 - ((time.time() - self.feedback_start_time) / self.feedback_duration)
            alpha = max(0.3, min(1.0, alpha))
            fb_h = 120
            fb_w = 560
            fb_x = (w - fb_w) // 2
            fb_y = (h - fb_h) // 2 - 40
            fb_overlay = frame.copy()
            cv2.rectangle(fb_overlay, (fb_x, fb_y), (fb_x + fb_w, fb_y + fb_h), (0, 170, 0), -1)
            cv2.addWeighted(fb_overlay, 0.88 * alpha, frame, 1 - 0.88 * alpha, 0, frame)
            cv2.rectangle(frame, (fb_x, fb_y), (fb_x + fb_w, fb_y + fb_h), (255, 255, 255), 3)
            frame = self._pil_to_cv2(pil_img) if False else frame
            pil_img = self._cv2_to_pil(frame)
            draw = ImageDraw.Draw(pil_img)
            tw, th = self._pil_text_size(draw, self.feedback_text, self.font_xl)
            tx = fb_x + (fb_w - tw) // 2
            ty = fb_y + (fb_h - th) // 2 - 2
            self._pil_text(draw, (tx, ty), self.feedback_text,
                           self.font_xl, (255, 255, 255), stroke=2)

        return self._pil_to_cv2(pil_img)

    def _draw_key_help(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        pil_img = self._cv2_to_pil(frame)
        draw = ImageDraw.Draw(pil_img)
        help_lines = [
            "按键说明：Q 或 ESC 退出   F 切换全屏   空格 暂停/恢复检测   R 录制新手势   ] 加亮  [ 减亮   . 加对比  , 减对比   0 重置",
            "当前映射：张开手掌 上一页   握拳 下一页   OK 从头播放   食指上 向上滚   剪刀 向下滚   六字(拇+小指) 返回上一页",
        ]
        for i, line in enumerate(reversed(help_lines)):
            self._pil_text(draw, (15, h - 10 - i * 22 - 2), line,
                           self.font_sm, (200, 200, 200))
        return self._pil_to_cv2(pil_img)

    def _resolve_action(self, gesture: str) -> Tuple[Optional[str], Optional[str]]:
        custom_key = self.bindings.get_key(gesture)
        if custom_key:
            return ACTION_CUSTOM, custom_key

        action = self.gesture_map.get(gesture)
        return action, None

    def _handle_recording(self, hand_landmarks, img_shape: Tuple[int, int], key: int) -> bool:
        now = time.time()

        if key == 27:
            self.recording_mode = False
            self.recorded_samples.clear()
            self.record_target_name = None
            self._set_feedback("已取消录制")
            return True

        if key == ord('r') or key == ord('R'):
            if len(self.recorded_samples) >= self.record_min_samples and self.record_target_name:
                self.recognizer.record_custom_gesture(self.record_target_name, self.recorded_samples)
                self._set_feedback(f"已保存手势: {self.record_target_name} ({len(self.recorded_samples)}个样本)")
                print(f"[录制] 手势 '{self.record_target_name}' 已保存, 样本数={len(self.recorded_samples)}")
                self.recording_mode = False
                self.recorded_samples.clear()
                self.record_target_name = None
            else:
                self._set_feedback(f"样本不足: {len(self.recorded_samples)}/{self.record_min_samples}")
            return True

        if hand_landmarks is not None:
            if (now - self.record_last_frame_time) > 0.15 and len(self.recorded_samples) < self.record_max_samples:
                lm = self.recognizer.extract_landmarks(hand_landmarks, img_shape)
                if lm:
                    self.recorded_samples.append(lm)
                    self.record_last_frame_time = now
        return False

    def _start_recording_flow(self):
        name = input("\n=== 录制新手势 ===\n请输入手势名称 (英文/拼音, 不含空格): ").strip()
        if not name:
            print("名称不能为空, 取消录制.")
            return
        if name in GESTURE_NAMES_CN or name in self.recognizer.list_custom_gestures():
            overwrite = input(f"手势 '{name}' 已存在, 是否覆盖? (y/n): ").strip().lower()
            if overwrite != 'y':
                print("已取消.")
                return

        key_str = input("请输入要绑定的按键 (单个键如 'f5', 或组合键如 'ctrl+s'). 直接回车跳过绑定: ").strip()

        self.record_target_name = name
        self.recorded_samples.clear()
        self.recording_mode = True
        self.record_last_frame_time = 0.0
        self._set_feedback(f"开始录制: {name} (保持手势稳定)")

        if key_str:
            self.bindings.bind(name, key_str)
            print(f"[绑定] '{name}' -> '{key_str}'")

    def run(self):
        self.running = True
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            print("错误：无法打开摄像头")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        prev_time = time.time()
        fps = 0.0
        fps_smooth = 0.0
        paused = False

        print("=" * 60)
        print(" 手势识别 PPT 控制系统 已启动")
        print("=" * 60)
        print(" 当前手势映射:")
        print("   张开手掌           -> 上一页 (Left Arrow)")
        print("   握拳               -> 下一页 (Right Arrow)")
        print("   OK 手势 (拇指食指相触) -> 从头播放 (Home)")
        print("   食指单独上指       -> 向上滚动 (PageUp)")
        print("   剪刀手势 (食+中指V字) -> 向下滚动 (PageDown)")
        print("   六字手势 (拇指+小指张开) -> 返回上一页 (Left Arrow)")
        print()
        print(" 操作说明:")
        print("   Q / ESC   - 退出程序")
        print("   F         - 全屏 / 窗口 切换")
        print("   空格      - 暂停/恢复防抖检测")
        print("   R         - 录制新手势 (在控制台输入信息)")
        print("   [ / ]     - 减小/增加亮度 (默认+35)")
        print("   , / .     - 减小/增加对比度 (默认1.2x)")
        print("   0         - 重置亮度/对比度/Gamma为默认")
        print()
        if self.recognizer.list_custom_gestures():
            print(f" 已录制的自定义手势: {self.recognizer.list_custom_gestures()}")
        print("=" * 60)

        cv2.namedWindow(self.WINDOW_TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW_TITLE, 960, 720)

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    print("摄像头读取失败")
                    break

                frame = cv2.flip(frame, 1)

                frame = self._adjust_brightness(frame)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('f') or key == ord('F'):
                    self._toggle_fullscreen()

                if key == ord('['):
                    self.brightness = max(-100, self.brightness - 10)
                    self._set_feedback(f"亮度: {int(self.brightness):+d}")
                elif key == ord(']'):
                    self.brightness = min(120, self.brightness + 10)
                    self._set_feedback(f"亮度: {int(self.brightness):+d}")
                elif key == ord(','):
                    self.contrast = max(0.5, round(self.contrast - 0.1, 1))
                    self._set_feedback(f"对比度: {self.contrast:.1f}x")
                elif key == ord('.'):
                    self.contrast = min(2.5, round(self.contrast + 0.1, 1))
                    self._set_feedback(f"对比度: {self.contrast:.1f}x")
                elif key == ord('0'):
                    self.brightness = 35
                    self.contrast = 1.2
                    self.gamma = 0.85
                    self._gamma_table = self._build_gamma_table(self.gamma)
                    self._set_feedback("已重置亮度/对比/Gamma")

                result = self.recognizer.process_frame(frame)
                hand_landmarks = None
                handedness_label = ""
                hand_exists = False

                if result.multi_hand_landmarks and len(result.multi_hand_landmarks) > 0:
                    hand_landmarks = result.multi_hand_landmarks[0]
                    hand_exists = True
                    if result.multi_handedness and len(result.multi_handedness) > 0:
                        handedness_label = result.multi_handedness[0].classification[0].label

                raw_lm = None
                if hand_landmarks is not None:
                    raw_lm = self.recognizer.extract_landmarks(hand_landmarks, frame.shape)
                    self.recognizer.draw_landmarks(frame, hand_landmarks)

                if key == ord('q') or key == 27:
                    if self.recording_mode:
                        self._handle_recording(hand_landmarks, frame.shape, 27)
                    else:
                        self._set_feedback("退出程序")
                        time.sleep(0.3)
                        break

                if key == ord('b') or key == ord('B'):
                    self._set_feedback("退出程序")
                    time.sleep(0.3)
                    break

                if key == ord(' '):
                    paused = not paused
                    self._set_feedback("已暂停检测" if paused else "已恢复检测")
                    self.debounce.reset()
                    continue

                if key == ord('r') or key == ord('R'):
                    if not self.recording_mode:
                        cv2.destroyAllWindows()
                        cap.release()
                        self._start_recording_flow()
                        cap = cv2.VideoCapture(self.camera_id)
                        if not cap.isOpened():
                            print("无法重新打开摄像头")
                            break
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        continue
                    else:
                        self._handle_recording(hand_landmarks, frame.shape, key)
                        continue

                if self.recording_mode:
                    self._handle_recording(hand_landmarks, frame.shape, -1)
                    detected_gesture = GESTURE_NONE
                    gesture_name = "录制中"
                else:
                    detected_gesture = self.recognizer.recognize(
                        hand_landmarks, handedness_label, raw_lm
                    )
                    if not paused:
                        stable_gesture, triggered, elapsed = self.debounce.update(detected_gesture)
                    else:
                        stable_gesture = detected_gesture
                        triggered = False
                    gesture_name = stable_gesture

                    if triggered:
                        action, custom_key = self._resolve_action(stable_gesture)
                        if action:
                            gesture_cn = GESTURE_NAMES_CN.get(stable_gesture, stable_gesture)
                            if action == ACTION_CUSTOM:
                                fb = f"已执行自定义按键: {custom_key}"
                                log_action_cn = f"自定义按键 {custom_key}"
                            else:
                                act_cn = ACTION_HINT_CN.get(action, f"执行动作 {action}")
                                fb = f"已执行: {act_cn}"
                                log_action_cn = act_cn
                            self._set_feedback(fb)
                            self.keyboard.execute_action(action, custom_key)
                            print(f"[手势触发] {gesture_cn}  →  {log_action_cn}")

                progress = 0.0 if paused or self.recording_mode else self.debounce.get_progress()

                now = time.time()
                instant_fps = 1.0 / max(1e-6, (now - prev_time))
                prev_time = now
                fps_smooth = fps_smooth * 0.9 + instant_fps * 0.1
                fps = fps_smooth

                frame = self._draw_hud(frame, gesture_name, progress, fps, hand_exists, handedness_label)
                frame = self._draw_key_help(frame)
                frame = self._fit_to_screen(frame)

                cv2.imshow(self.WINDOW_TITLE, frame)

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.running = False
            print("程序已退出, 资源已释放.")


def main():
    controller = GesturePPTController(
        camera_id=0,
        debounce_time=1.0,
        target_fps=30,
        show_fps=True
    )
    controller.run()


if __name__ == "__main__":
    main()
