import os
import json
from PIL import Image

images_path = "dataset/train/images"
labels_path = "dataset/train/labels"

coco = {
    "images": [],
    "annotations": [],
    "categories": []
}

category_set = set()
annotation_id = 0
image_id = 0

for filename in os.listdir(images_path):
    if not filename.endswith(".jpg"):
        continue

    image_path = os.path.join(images_path, filename)
    label_path = os.path.join(labels_path, filename.replace(".jpg", ".txt"))

    if not os.path.exists(label_path):
        continue

    # Get image size
    img = Image.open(image_path)
    width, height = img.size

    coco["images"].append({
        "id": image_id,
        "file_name": filename,
        "width": width,
        "height": height
    })

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])

        # Convert YOLO → COCO bbox
        x = (x_center - w / 2) * width
        y = (y_center - h / 2) * height
        bbox_width = w * width
        bbox_height = h * height

        coco["annotations"].append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": class_id,
            "bbox": [x, y, bbox_width, bbox_height],
            "area": bbox_width * bbox_height,
            "iscrowd": 0
        })

        category_set.add(class_id)
        annotation_id += 1

    image_id += 1

# Add categories
for cat_id in sorted(category_set):
    coco["categories"].append({
        "id": cat_id,
        "name": f"class_{cat_id}"
    })

# Save JSON
with open("train_annotations.json", "w") as f:
    json.dump(coco, f, indent=4)

print("✅ COCO file created successfully!")