"""
The way dataset was organized, the train set had 1K folders where each folder corresponded to a class. The validation folder only had images without class information. The class information was in /ILSVRC/LOC_val_solution_cls.csv. Hence, this script just makes the validation data consistent with the format in train data.
"""
import os
import shutil
import csv

# Validation set
val_dir = "/ssd_scratch/alexnet/ILSVRC/Data/CLS-LOC/val"  

# Labels for Validation set
csv_file = "/ssd_scratch/alexnet/LOC_val_solution_cls.csv"

image_to_class = {}
with open(csv_file, 'r') as f:
    reader = csv.reader(f)
    next(reader)  
    for row in reader:
        img_name, class_id = row
        image_to_class[img_name] = class_id

for img_file in os.listdir(val_dir):
    # All images are JPEG
    if not img_file.lower().endswith((".jpeg")):
        continue

    img_name = os.path.splitext(img_file)[0] 
    if img_name not in image_to_class:
        print(f"{img_name} not found in CSV")
        break

    class_id = image_to_class[img_name]
    class_dir = os.path.join(val_dir, class_id)
    os.makedirs(class_dir, exist_ok=True)

    src = os.path.join(val_dir, img_file)
    dst = os.path.join(class_dir, img_file)
    shutil.move(src, dst)

print("Validation set is reorganized into 1K class folders")