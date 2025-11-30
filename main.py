import os
import argparse
import random
from PIL import Image, ImageDraw


CLASSES = ["circle", "square", "triangle"]


def random_color():
    return tuple(random.randint(50, 255) for _ in range(3))


def generate_shape_image(shape, size=224):
    bg_color = (0, 0, 0)  # sfondo nero
    img = Image.new("RGB", (size, size), bg_color)
    draw = ImageDraw.Draw(img)

    padding = size // 8

    x1 = random.randint(padding, size // 2)
    y1 = random.randint(padding, size // 2)
    x2 = random.randint(size // 2, size - padding)
    y2 = random.randint(size // 2, size - padding)

    color = random_color()

    if shape == "circle":
        draw.ellipse([x1, y1, x2, y2], fill=color)

    elif shape == "square":
        side = min(x2 - x1, y2 - y1)
        x2 = x1 + side
        y2 = y1 + side
        draw.rectangle([x1, y1, x2, y2], fill=color)

    elif shape == "triangle":
        points = [
            (random.randint(padding, size - padding),
             random.randint(padding, size - padding))
            for _ in range(3)
        ]
        draw.polygon(points, fill=color)

    return img


def generate_split(split_dir, num_per_class, size):
    os.makedirs(split_dir, exist_ok=True)
    for cls in CLASSES:
        class_dir = os.path.join(split_dir, cls)
        os.makedirs(class_dir, exist_ok=True)
        for i in range(num_per_class):
            img = generate_shape_image(cls, size=size)
            img_path = os.path.join(class_dir, f"{cls}_{i:04d}.png")
            img.save(img_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="data224")
    parser.add_argument("--train_per_class", type=int, default=500)
    parser.add_argument("--val_per_class", type=int, default=100)
    parser.add_argument("--size", type=int, default=224)
    args = parser.parse_args()

    train_dir = os.path.join(args.out_dir, "train")
    val_dir = os.path.join(args.out_dir, "val")

    print(f"Genero TRAIN {args.size}×{args.size}...")
    generate_split(train_dir, args.train_per_class, args.size)

    print(f"Genero VAL {args.size}×{args.size}...")
    generate_split(val_dir, args.val_per_class, args.size)

    print("Fatto.")


if __name__ == "__main__":
    main()

