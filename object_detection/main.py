from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import torch

# Load model + processor
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained(
    "facebook/detr-resnet-50"
)

# Load image
image = Image.open(r"C:\Users\ANSHIKA bapna\OneDrive\Desktop\internship\object_detection\image\text.jpg")

# Preprocess
inputs = processor(images=image, return_tensors="pt")

# Inference
outputs = model(**inputs)

# Post-process
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(
    outputs, target_sizes=target_sizes, threshold=0.5
)[0]

# Print results
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    print(f"Detected {model.config.id2label[label.item()]} with confidence {score.item():.2f}")