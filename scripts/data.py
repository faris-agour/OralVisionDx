import os
import shutil
import splitfolders
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA = PROJECT_ROOT / "data" / "raw" / "oral-diseases"
OUT_DATA = PROJECT_ROOT / "data" / "clean_dataset"
FINAL_DATASET = PROJECT_ROOT / "data" / "dataset"

CLASSES = {
    "Caries": "Caries",
    "Calculus": "Calculus",
    "Gingivitis": "Gingivitis",
    "Mouth Ulcer": "Ulcer",
    "Tooth Discoloration": "Discoloration",
    "hypodontia": "Hypodontia"
}

os.makedirs(str(OUT_DATA), exist_ok=True)

def collect_images(src, dst_folder):
    os.makedirs(dst_folder, exist_ok=True)
    counter = len(os.listdir(dst_folder))
    for root, dirs, files in os.walk(src):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                src_file = os.path.join(root, f)
                dst_file = os.path.join(dst_folder, f"{counter}_{f}")
                shutil.copy2(src_file, dst_file)
                counter += 1

print("Starting dataset cleanup\n")

for folder in os.listdir(str(RAW_DATA)):
    full_path = os.path.join(str(RAW_DATA), folder)
    for key in CLASSES:
        if key.lower().replace(" ", "") in folder.lower().replace(" ", ""):
            class_name = CLASSES[key]
            print(f"> Collecting: {class_name}  from  {folder}")
            collect_images(full_path, os.path.join(str(OUT_DATA), class_name))

print("\nSplitting dataset 80% / 10% / 10%")
splitfolders.ratio(str(OUT_DATA),
                   output=str(FINAL_DATASET),
                   seed=42, ratio=(.8, .1, .1))
print("DONE!")
