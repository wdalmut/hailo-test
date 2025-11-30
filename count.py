import os

base = "data224/train"
for cls in os.listdir(base):
    cpath = os.path.join(base, cls)
    if os.path.isdir(cpath):
        n = len([f for f in os.listdir(cpath) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        print("train", cls, "->", n)

base = "data224/val"
for cls in os.listdir(base):
    cpath = os.path.join(base, cls)
    if os.path.isdir(cpath):
        n = len([f for f in os.listdir(cpath) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        print("val", cls, "->", n)
