import json
import cv2
import matplotlib.pyplot as plt

# load coco file
with open("train_annotations.json") as f:
    data = json.load(f)

# pick first image
img_info = data["images"][0]

# correct path
img_path = "dataset/train/images/" + img_info["file_name"]

# read image
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# draw bounding boxes
for ann in data["annotations"]:
    if ann["image_id"] == img_info["id"]:
        x, y, w, h = ann["bbox"]
        cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), (255, 0, 0), 2)

# show image
plt.imshow(img)
plt.title("Bounding Box Check")
plt.axis("off")
plt.show()