import cv2
import numpy as np
import re
from keras import ops
import keras
import keras_hub

# 推論精度設定（bfloat16：省メモリ）
keras.config.set_floatx("bfloat16")

# モデルロード（初回は時間かかります）
paligemma = keras_hub.models.PaliGemmaCausalLM.from_preset(
    "kaggle://keras/paligemma2/keras/pali_gemma2_mix_3b_224"
)


# ボックスとラベルを抽出する関数
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


# --- 推論対象の画像ファイルを指定 ---
image_path = "car.jpg"  # JPEG画像を用意して同ディレクトリに置いてください
target_size = (224, 224)

# 画像読み込み & 前処理
frame = cv2.imread(image_path)
if frame is None:
    raise ValueError("画像が読み込めませんでした。ファイルパスを確認してください。")

h, w, _ = frame.shape
side = min(h, w)
center_y = h // 2
center_x = w // 2
top = center_y - side // 2
left = center_x - side // 2
cropped = frame[top : top + side, left : left + side]
resized = cv2.resize(cropped, target_size)
rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

# Tensorに変換（1枚のバッチにする）
rgb_tensor = ops.convert_to_tensor(rgb, dtype="uint8")
rgb_tensor = ops.expand_dims(rgb_tensor, 0)

# --- 検出実行 ---
prompt = "detect cow"
print("Generating...")
output = paligemma.generate(inputs={"images": rgb_tensor, "prompts": [prompt]})
print("Generated!")

# 出力の中身を確認（オプション）
print("Raw output:\n", output)

# ボックス・ラベル抽出
boxes, labels = parse_bbox_and_labels(output[0])

# 結果描画
for box, label in zip(boxes, labels):
    y0, x0, y1, x1 = (box * target_size[0]).astype(int)
    cv2.rectangle(resized, (x0, y0), (x1, y1), (0, 255, 0), 2)
    cv2.putText(
        resized, label, (x0, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
    )

cv2.imshow("Detection Result", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
