from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import torch
import matplotlib.pyplot as plt


processor = DetrImageProcessor.from_pretrained("my_model")
model = DetrForObjectDetection.from_pretrained("my_model")

image = Image.open("dataset/train/images/00095_176.jpg")

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)


target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]

plt.imshow(image)
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    if score > 0.5:
        x1, y1, x2, y2 = box.tolist()
        plt.gca().add_patch(plt.Rectangle((x1,y1), x2-x1, y2-y1, fill=False, color='red', linewidth=2))
plt.show()