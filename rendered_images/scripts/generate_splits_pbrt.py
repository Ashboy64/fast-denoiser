import os
import random
import numpy as np


DATASET_PATH = "../pbrt/san_miguel"

SEED = 1

TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
TEST_FRAC = 0.1

assert TRAIN_FRAC + VAL_FRAC + TEST_FRAC == 1.0


ALL_SPP = [1, 4, 8, 1024]
TOTAL_NUM_SAMPLES = 1600


def write_to_file(samples, file_path):
    if os.path.exists(file_path):
        os.remove(file_path)

    with open(file_path, mode="x") as f:
        f.write("\n".join(samples))


def generate_split():
    num_test = int(TEST_FRAC * TOTAL_NUM_SAMPLES)
    num_val = int(VAL_FRAC * TOTAL_NUM_SAMPLES)
    
    file_prefixes = [f"random_camera_{i + 1}" for i in range(TOTAL_NUM_SAMPLES)]
    np.random.shuffle(file_prefixes)

    test_samples = file_prefixes[: num_test]
    val_samples = file_prefixes[num_test : num_test + num_val]
    train_samples = file_prefixes[num_test + num_val :]

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
