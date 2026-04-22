import os
import shutil

base_path = "dataset/images"

for split in ["train", "test"]:
    split_path = os.path.join(base_path, split)

    images_out = os.path.join("dataset", split, "images")
    labels_out = os.path.join("dataset", split, "labels")

    os.makedirs(images_out, exist_ok=True)
    os.makedirs(labels_out, exist_ok=True)

    for file in os.listdir(split_path):
        src = os.path.join(split_path, file)

        if file.endswith(".jpg") or file.endswith(".png"):
            shutil.move(src, os.path.join(images_out, file))
        elif file.endswith(".txt"):
            shutil.move(src, os.path.join(labels_out, file))

print("✅ Files organized successfully!")