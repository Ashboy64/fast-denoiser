import os
import random
import numpy as np


# DATASET_PATH = "../blender/classroom/unzipped"
DATASET_PATH = "../blender/amazon_bistro/unzipped"

SEED = 0

TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
TEST_FRAC = 0.1

assert TRAIN_FRAC + VAL_FRAC + TEST_FRAC == 1.0


def get_batch_dirs():
    return [
        dirname
        for dirname in os.listdir(DATASET_PATH)
        if os.path.isdir(os.path.join(DATASET_PATH, dirname))
    ]


def write_to_file(samples, file_path):
    if os.path.exists(file_path):
        os.remove(file_path)

    with open(file_path, mode="x") as f:
        f.write("\n".join(samples))


def generate_split():

    batch_dirs = get_batch_dirs()

    train_samples = []
    val_samples = []
    test_samples = []

    for raw_batch_dir in batch_dirs:
        batch_dir = os.path.join(DATASET_PATH, raw_batch_dir, "samples_1")

        filenames = [
            f"{raw_batch_dir},{filename}"
            for filename in os.listdir(batch_dir)
            if os.path.isfile(os.path.join(batch_dir, filename))
            and filename not in [".DS_STORE", ".DS_Store"]
        ]

        np.random.shuffle(filenames)

        num_samples = len(filenames)

        num_test = int(TEST_FRAC * num_samples)
        num_val = int(VAL_FRAC * num_samples)

        test_samples = test_samples + filenames[:num_test]
        val_samples = val_samples + filenames[num_test : num_test + num_val]
        train_samples = train_samples + filenames[num_test + num_val :]

    write_to_file(train_samples, os.path.join(DATASET_PATH, "train_split.txt"))
    write_to_file(val_samples, os.path.join(DATASET_PATH, "val_split.txt"))
    write_to_file(test_samples, os.path.join(DATASET_PATH, "test_split.txt"))

    print(f"SAVED SPLIT")
    print(f"num_train = {len(train_samples)}")
    print(f"num_val = {len(val_samples)}")
    print(f"num_test = {len(test_samples)}")


def seed(seed_val):
    random.seed(seed_val)
    np.random.seed(seed_val)


if __name__ == "__main__":
    seed(SEED)
    generate_split()
