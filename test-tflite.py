import tensorflow as tf
import numpy as np
from PIL import Image

IMG_SIZE = 224
TFLITE_PATH = "model/model.tflite"  # dove l'hai salvato

class_names = ["circle", "square", "triangle"]

interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("input_details:", input_details)
print("output_details:", output_details)

def load_img(path):
    img = Image.open(path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    x = np.array(img, dtype=np.float32)
    # ⚠️ niente /255 qui, c'è sempre il layer Rescaling nel grafo
    x = np.expand_dims(x, axis=0)
    return x

test_paths = [
    "data224/val/circle/circle_0071.png",
    "data224/val/square/square_0071.png",
    "data224/val/triangle/triangle_0071.png",
]

for p in test_paths:
    x = load_img(p)
    interpreter.set_tensor(input_details[0]["index"], x)
    interpreter.invoke()
    y = interpreter.get_tensor(output_details[0]["index"])[0]  # (3,)
    cid = int(np.argmax(y))
    print("TFLite", p, "->", class_names[cid], y)

