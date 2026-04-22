import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DetrImageProcessor, DetrForObjectDetection


with open("train_annotations.json") as f:
    coco = json.load(f)

images = coco["images"]
annotations = coco["annotations"]


ann_dict = {}
for ann in annotations:
    ann_dict.setdefault(ann["image_id"], []).append(ann)


class CocoDataset(Dataset):
    def __init__(self, images, ann_dict, img_dir):
        self.images = images
        self.ann_dict = ann_dict
        self.img_dir = img_dir

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = os.path.join(self.img_dir, img_info["file_name"])
        
        image = Image.open(img_path).convert("RGB")

        anns = self.ann_dict.get(img_info["id"], [])

        boxes = []
        labels = []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x+w, y+h])
            labels.append(ann["category_id"])

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)
        }

        return image, target


dataset = CocoDataset(images, ann_dict, "dataset/train/images")
def collate_fn(batch):
    images = [item[0] for item in batch]   # list of PIL images
    targets = [item[1] for item in batch]  # list of dicts
    return images, targets
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)


processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained(
    "facebook/detr-resnet-50",
    num_labels=2,
    ignore_mismatched_sizes=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


model.train()

for epoch in range(2):  
    for images, targets in dataloader:
        inputs = processor(images=images, return_tensors="pt").to(device)

        labels = []
        for t in targets:
            labels.append({
                "class_labels": t["labels"].to(device),
                "boxes": t["boxes"].to(device)
            })

        outputs = model(**inputs, labels=labels)
        loss = outputs.loss

        loss.backward()
        print("Loss:", loss.item())
        
model.save_pretrained("my_model")
processor.save_pretrained("my_model")

print("✅ Model saved successfully!")