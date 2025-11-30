import tensorflow as tf
import keras
import numpy as np
from PIL import Image
import os

IMG_SIZE = 224
SAVEDMODEL_DIR = "model/1"  # la tua cartella esportata con model.export()

class_names = ["circle", "square", "triangle"]

# Carico il SavedModel come layer di inferenza
layer = keras.layers.TFSMLayer(SAVEDMODEL_DIR, call_endpoint="serving_default")

def load_img(path):
    img = Image.open(path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    x = np.array(img, dtype=np.float32)
    # ⚠️ niente /255: lo fa già il layer Rescaling nel modello
    x = np.expand_dims(x, axis=0)  # (1, 224, 224, 3)
    return x

test_paths = [
    "data224/val/circle/circle_0071.png",
    "data224/val/square/square_0071.png",
    "data224/val/triangle/triangle_0071.png",
]

for p in test_paths:
    x = load_img(p)
    # TFSMLayer restituisce un dict di output -> prendiamo il primo
    out = layer(x)
    # se non sai la chiave:
    # print(out.keys())
    y = list(out.values())[0].numpy()[0]   # vettore (3,)
    cid = int(np.argmax(y))
    print(p, "->", class_names[cid], y)
