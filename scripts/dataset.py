import os
import shutil
import kagglehub
import numpy as np
import pandas as pd
from PIL import Image


class Dataset:
    emotion_labels = {}

    def __init__(self, name, path):
        self.name = name
        self.path = path
        self.downloaded = False
        self.download_path = os.path.join(os.path.dirname(__file__), "..", "temporal_datasets", self.name)
        self.post_processed_path = os.path.join(os.path.dirname(__file__), "..", "datasets", self.name)

    def __repr__(self):
        return f"Dataset(name={self.name}, path={self.path})"

    def download_and_move(self):
        if os.path.exists(self.download_path):
            shutil.rmtree(os.path.join(self.download_path))

        temporal_path = kagglehub.dataset_download(self.path)
        self.downloaded = True

        os.makedirs(self.download_path, exist_ok=True)
        shutil.move(temporal_path, self.download_path)

    def post_processing(self):
        """
        Post-processing method to be implemented by subclasses.
        This method should handle any dataset-specific post-processing tasks.
        """
        self.create_dir_structure()
        self.generate_images()

    def create_dir_structure(self):
        if not os.path.exists(self.post_processed_path):
            os.makedirs(self.post_processed_path)

        for usage in ["train", "valid", "test"]:
            usage_folder_path = os.path.join(self.post_processed_path, usage)
            if not os.path.exists(usage_folder_path):
                os.makedirs(usage_folder_path)
            for label in self.emotion_labels.values():
                subfolder_path = os.path.join(usage_folder_path, label)
                if not os.path.exists(subfolder_path):
                    os.makedirs(subfolder_path)

    def generate_images(self):
        df = pd.read_csv(os.path.join(self.download_path, self.csv_location))

        for index, row in df.iterrows():
            # Extract the image data from the row
            pixels = row["pixels"].split()
            img_data = [int(pixel) for pixel in pixels]
            img_array = np.array(img_data).reshape(48, 48)
            img = Image.fromarray(img_array.astype("uint8"), "L")

            # Get the emotion label and determine the output subfolder based on the Usage column
            emotion_label = self.emotion_labels.get(str(row["emotion"]), None)
            if emotion_label is None:
                continue
            if row["Usage"] == "Training":
                output_subfolder_path = os.path.join(self.post_processed_path, "train", emotion_label)
            elif row["Usage"] == "PublicTest":
                output_subfolder_path = os.path.join(self.post_processed_path, "valid", emotion_label)
            else:
                output_subfolder_path = os.path.join(self.post_processed_path, "test", emotion_label)

            # Save the image to the output subfolder
            output_file_path = os.path.join(output_subfolder_path, f"{index}.jpg")
            img.save(output_file_path)

    def delete_downloaded_files(self):
        if self.downloaded and self.download_path:
            shutil.rmtree(self.download_path)
