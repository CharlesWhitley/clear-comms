import os
import random
import shutil

# paths
DATASET_DIR = "dataset"
GAMEPLAY_DIR = os.path.join(DATASET_DIR, "gameplay_only")
MIXED_DIR = os.path.join(DATASET_DIR, "mixed")

TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")

# ratio for training vs validation
TRAIN_RATIO = 0.8

def make_dirs():
    for path in [TRAIN_DIR, VAL_DIR]:
        for sub in ["gameplay_only", "mixed"]:
            os.makedirs(os.path.join(path, sub), exist_ok=True)

def split_dataset():
    # get all files from gameplay_only 
    files = os.listdir(GAMEPLAY_DIR)
    files = [f for f in files if f.endswith(".wav")]

    random.shuffle(files)

    train_cutoff = int(len(files) * TRAIN_RATIO)
    train_files = files[:train_cutoff]
    val_files = files[train_cutoff:]

    # copy files into train/val dirs
    for fname in train_files:
        shutil.copy(os.path.join(GAMEPLAY_DIR, fname),
                    os.path.join(TRAIN_DIR, "gameplay_only", fname))
        shutil.copy(os.path.join(MIXED_DIR, fname),
                    os.path.join(TRAIN_DIR, "mixed", fname))

    for fname in val_files:
        shutil.copy(os.path.join(GAMEPLAY_DIR, fname),
                    os.path.join(VAL_DIR, "gameplay_only", fname))
        shutil.copy(os.path.join(MIXED_DIR, fname),
                    os.path.join(VAL_DIR, "mixed", fname))

    print(f"✅ Split complete: {len(train_files)} train files, {len(val_files)} val files")

if __name__ == "__main__":
    make_dirs()
    split_dataset()
