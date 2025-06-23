import os
import random
import shutil
from dataset import Dataset


class RafDB(Dataset):
    emotion_labels = {
        "1": "Surprise",
        "2": "Fear",
        "3": "Disgust",
        "4": "Happy",
        "5": "Sad",
        "6": "Angry",
        "7": "Neutral",
    }

    def __init__(self):
        super().__init__("rafdb", "shuvoalok/raf-db-dataset")

    def post_processing(self):
        self.create_dir_structure()
        self.distribute_images()

    def distribute_images(self):
        raw_dataset = os.path.join(self.download_path, '2/DATASET')
        for key, label in self.emotion_labels.items():
            train_files = os.path.join(raw_dataset, "train", key)
            test_files = os.path.join(raw_dataset, "test", key)
            total_files = []
            total_files.extend([os.path.join(train_files, f) for f in os.listdir(train_files)])
            total_files.extend([os.path.join(test_files, f) for f in os.listdir(test_files)])

            total_files = list(set(total_files))  # Remove duplicates if any
            random.shuffle(total_files)
            n_total = len(total_files)
            n_train = int(0.8 * n_total)
            n_valid = int(0.1 * n_total)

            train_selected_files = total_files[:n_train]
            valid_selected_files = total_files[n_train:n_train + n_valid]
            test_selected_files = total_files[n_train + n_valid:]

            for f in train_selected_files:
                shutil.move(f, os.path.join(self.post_processed_path, "train", label))

            for f in valid_selected_files:
                shutil.move(f, os.path.join(self.post_processed_path, "valid", label))

            for f in test_selected_files:
                shutil.move(f, os.path.join(self.post_processed_path, "test", label))
