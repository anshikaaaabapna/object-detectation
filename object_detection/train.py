from pylabel import importer

dataset = importer.ImportYoloV5(
    path="dataset/train",  
    path_to_images="dataset/train/images"
)

print("Annotations loaded:", len(dataset.df))

if len(dataset.df) > 0:
    dataset.export.ExportToCoco("train_annotations.json")
    print("✅ Conversion successful!")
else:
    print("❌ No annotations found")