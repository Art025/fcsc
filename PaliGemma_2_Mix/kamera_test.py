import cv2

cap = cv2.VideoCapture("/dev/video0")  # ← ここを1に変える
if not cap.isOpened():
    print("カメラが開けません")
else:
    ret, frame = cap.read()
    if ret:
        cv2.imshow("Camera", frame)
        cv2.waitKey(0)
    else:
        print("フレームが取得できませんでした")
cap.release()
cv2.destroyAllWindows()
