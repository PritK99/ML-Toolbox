"""
The images in train folder are organized as per their class
But the images in val folder are not organized as per their class
We need to organize val folder to make it compatible with torchvision
"""
import os
import shutil
import xml.etree.ElementTree as ET

val_dir = "ILSVRC/Data/CLS-LOC/val"
val_annotations_dir = "ILSVRC/Annotations/CLS-LOC/val"
val_images = os.listdir(val_dir)

print(f"Number of val images: {len(val_images)}")

for val_img in val_images:
    val_img_path = os.path.join(val_dir, val_img)
    val_annotation_filename = val_img[:-5] + ".xml"
    val_annotation_file = os.path.join(val_annotations_dir, val_annotation_filename)

    tree = ET.parse(val_annotation_file)
    synset = tree.getroot().find("object").find("name").text    # This is the class of the objct

    synset_dir_path = os.path.join(val_dir, synset) 
    os.makedirs(synset_dir_path, exist_ok=True)

    shutil.move(val_img_path, synset_dir_path)

print(f"Number of val images after moving: {len(val_images)}")