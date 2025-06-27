import cv2
from ultralytics import YOLO


def visualize_bbox(frame, bbox, label=None, color=(0, 255, 0), thickness=2):
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    if label:
        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            thickness,
        )
    return frame


def main():
    model = YOLO("yolo11n.pt").eval()
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        for result in results:
            for bbox in result.boxes.xyxy.cpu().numpy():
                label = result.names[int(result.boxes.cls.cpu().numpy()[0])]
                frame = visualize_bbox(frame, bbox, label)

        cv2.imshow("YOLO Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()


if __name__ == "__main__":
    main()
