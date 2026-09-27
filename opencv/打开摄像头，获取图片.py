import cv2

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("无法打开摄像头")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # ===== 处理与显示 =====
        cv2.imshow("Camera", frame)

        # 关闭方式: 按 q 键
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
