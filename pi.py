import time
import cv2
import numpy as np

from picamera2 import Picamera2

from hailo_platform import (
    HEF,
    VDevice,
    Device,
    ConfigureParams,
    InputVStreamParams,
    OutputVStreamParams,
    InputVStreams,
    OutputVStreams,
    HailoStreamInterface,
)

# === CONFIG ===
HEF_PATH = "/home/pi/hailo-test/model/shapes224.hef"  # <-- metti qui il tuo .hef
THRESH = 0.3  # soglia di confidenza

# nomi delle classi nel tuo modello (ordina come in training)
CLASS_NAMES = ['circle', 'square', 'triangle']


def decode_detections(dets_tensor, img_w, img_h, score_thresh=0.3):
    """
    Assumo che dets_tensor sia shape (N, 6):
      [x1_norm, y1_norm, x2_norm, y2_norm, score, class_id]
    dove coordinate sono normalizzate in [0, 1].
    Ritorna una lista di dict:
      {"x1":..., "y1":..., "x2":..., "y2":..., "score":..., "label":...}
    """
    if isinstance(dets_tensor, list):
        dets_tensor = np.array(dets_tensor)

    if dets_tensor.size == 0:
        return []

    dets_tensor = dets_tensor.reshape(-1, dets_tensor.shape[-1])
    results = []

    for det in dets_tensor:
        # se non hai class_id (solo 5 valori) togli la parte del class_id
        if det.shape[0] < 6:
            # [x1n, y1n, x2n, y2n, score] senza class_id
            x1n, y1n, x2n, y2n, score = det.astype(float)
            class_id = 0
        else:
            x1n, y1n, x2n, y2n, score, class_id = det.astype(float)

        if score < score_thresh:
            continue

        # da coordinate normalizzate a pixel
        x1 = int(x1n * img_w)
        y1 = int(y1n * img_h)
        x2 = int(x2n * img_w)
        y2 = int(y2n * img_h)

        # clamp nei limiti immagine
        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(0, min(x2, img_w - 1))
        y2 = max(0, min(y2, img_h - 1))

        class_id = int(class_id)
        if 0 <= class_id < len(CLASS_NAMES):
            label = CLASS_NAMES[class_id]
        else:
            label = f"id_{class_id}"

        results.append(
            {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "score": score,
                "label": label,
            }
        )

    return results


def main():
    # --- Carico il modello e leggo la dimensione di input ---
    hef = HEF(HEF_PATH)
    input_info = hef.get_input_vstream_infos()[0]
    in_h, in_w = input_info.shape[0], input_info.shape[1]
    print(f"Input model size: {in_w}x{in_h}")

    # --- Inizializzo la camera (Picamera2) ---
    picam = Picamera2()

    video_config = picam.create_video_configuration(
        main={"size": (in_w, in_h), "format": "RGB888"}
    )
    picam.configure(video_config)
    picam.start()
    time.sleep(0.5)

    # --- Trovo i device Hailo ---
    devices = Device.scan()
    if not devices:
        raise RuntimeError("Nessun dispositivo Hailo trovato (Device.scan() vuoto)")
    print("Dispositivi Hailo:", devices)

    # --- Creo VDevice e configuro il network group ---
    with VDevice(device_ids=devices) as target:
        cfg_params = ConfigureParams.create_from_hef(
            hef,
            interface=HailoStreamInterface.PCIe,  # M.2 su Pi in genere è PCIe
        )
        network_groups = target.configure(hef, cfg_params)
        network_group = network_groups[0]
        network_group_params = network_group.create_params()

        input_vstreams_params = InputVStreamParams.make_from_network_group(network_group)
        output_vstreams_params = OutputVStreamParams.make_from_network_group(network_group)

        with InputVStreams(network_group, input_vstreams_params) as in_streams, \
             OutputVStreams(network_group, output_vstreams_params) as out_streams, \
             network_group.activate(network_group_params):

            print("Inferenza modello custom avviata. Premi 'q' per uscire.")

            # opzionale: stampa info sugli stream
            print("--- Input streams ---")
            for s in in_streams:
                print("  ", s.name)
            print("--- Output streams ---")
            for s in out_streams:
                print("  ", s.name)

            while True:
                # Frame dalla camera (RGB)
                frame_rgb = picam.capture_array()

                # Il modello si aspetta (1, H, W, C) uint8
                input_tensor = np.expand_dims(frame_rgb, axis=0).astype(np.uint8)

                # manda l'input a tutti gli stream (in genere 1)
                for s in in_streams:
                    s.send(input_tensor)

                # raccogli tutti i tensori di output
                outputs = []
                for s in out_streams:
                    out = s.recv()
                    outputs.append(out)

                # Se hai un solo tensore di detection:
                dets_tensor = outputs[0]

                # decodifica le detection in bbox + label + score
                detections = decode_detections(
                    dets_tensor,
                    img_w=in_w,
                    img_h=in_h,
                    score_thresh=THRESH,
                )

                # frame_bgr per disegnare con OpenCV
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                # disegna le box
                for det in detections:
                    x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
                    score = det["score"]
                    label = det["label"]

                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    text = f"{label} {score:.2f}"
                    cv2.putText(
                        frame_bgr,
                        text,
                        (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )

                cv2.imshow("Modello custom Hailo - preview", frame_bgr)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    picam.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

