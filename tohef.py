from pathlib import Path
import numpy as np
from PIL import Image

from hailo_sdk_client import ClientRunner


TFLITE_MODEL = "model/model.tflite"
CALIB_DIR = "./data224"   # cartella con PNG/JPG 224x224
OUTPUT_HAR = "model_h8l.har"
MODEL_NAME = "shapes224"     # nome arbitrario per il modello

def load_calib_images_list(calib_dir, img_size=(224, 224)):
    paths = list(Path(calib_dir).glob("**/*.png")) + list(Path(calib_dir).glob("**/*.jpg"))
    if not paths:
        raise RuntimeError(f"Nessuna immagine trovata in {calib_dir}")
    print(f"Trovate {len(paths)} immagini di calibrazione")

    images = []
    for p in paths:
        img = Image.open(p).convert("RGB").resize(img_size)
        arr = np.array(img, dtype=np.float32)
        arr = arr / 255.0  # se in training hai Rescaling(1./255)
        images.append(arr)

    # shape: (N, H, W, C)
    calib_array = np.stack(images, axis=0)
    print("Shape calibrazione:", calib_array.shape, calib_array.dtype)
    return calib_array

def main():
    # 1) Crea il runner per Hailo8L
    runner = ClientRunner(hw_arch="hailo8l")

    # 2) PARSING: traduce il modello TFLite nel formato Hailo
    #   (alcune versioni usano translate_tf_model anche per i .tflite)
    print(f"Traduco il modello TFLite: {TFLITE_MODEL}")
    hn, npz = runner.translate_tf_model(
        TFLITE_MODEL,
        MODEL_NAME,
    )

    # 3) OPTIMIZATION (quantizzazione & co.) con dataset di calibrazione
    print("Avvio ottimizzazione con dataset di calibrazione...")
    calib_gen = load_calib_images_list(CALIB_DIR, img_size=(224, 224))
    runner.optimize(calib_gen)

    # 4) COMPILAZIONE → HAR per hailo8l
    print("Compilo il modello per Hailo8L...")
    runner.compile()

    # 5) Salva HAR
    runner.save_har(OUTPUT_HAR)
    print(f"HAR generato: {OUTPUT_HAR}")


if __name__ == "__main__":
    main()

