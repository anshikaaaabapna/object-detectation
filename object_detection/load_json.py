import json

with open("train_annotations.json") as f:
    data = json.load(f)

print("Total Images:", len(data["images"]))
print("Total Annotations:", len(data["annotations"]))
print("Categories:", data["categories"])