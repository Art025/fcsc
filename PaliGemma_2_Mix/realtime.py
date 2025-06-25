import cv2
import keras
import keras_hub
import numpy as np
import re
import time


keras.config.set_floatx("bfloat16")

paligemma = keras_hub.models.PaliGemmaCausalLM.from_preset(
    "kaggle://keras/paligemma2/keras/pali_gemma2_mix_3b_224"
)


# 検出結果のパース用
def parse_bbox_and_labels(detokenized_output: str):
    matches = re.finditer(
        "<loc(?P<y0>\d\d\d\d)><loc(?P<x0>\d\d\d\d)><loc(?P<y1>\d\d\d\d)><loc(?P<x1>\d\d\d\d)>"
        " (?P<label>.+?)( ;|$)",
        detokenized_output,
    )
    labels, boxes = [], []
    fmt = lambda x: float(x) / 1024.0
    for m in matches:
        d = m.groupdict()
        boxes.append([fmt(d["y0"]), fmt(d["x0"]), fmt(d["y1"]), fmt(d["x1"])])
        labels.append(d["label"])
    return np.array(boxes), np.array(labels)


# カメラ初期化
cap = cv2.VideoCapture("/dev/video0")
target_size = (224, 224)

last_inference_time = 0
inference_interval = 1.0  # 秒

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 正方形中央トリミング＆リサイズ
    h, w, _ = frame.shape
    side = min(h, w)
    center_y = h // 2
    center_x = w // 2
    top = center_y - side // 2
    left = center_x - side // 2
    cropped = frame[top : top + side, left : left + side]
    resized = cv2.resize(cropped, target_size)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # 検出プロンプト
    prompt = "detect cow\n"
    output = paligemma.generate(inputs={"images": rgb, "prompts": prompt})

    # 出力からボックス抽出
    boxes, labels = parse_bbox_and_labels(output)

    # 検出結果を描画
    for box, label in zip(boxes, labels):
        y0, x0, y1, x1 = (box * target_size[0]).astype(int)
        cv2.rectangle(resized, (x0, y0), (x1, y1), (0, 0, 255), 2)
        cv2.putText(
            resized, label, (x0, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
        )

    cv2.imshow("PaliGemma Detection", resized)

    # qキーで終了
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
