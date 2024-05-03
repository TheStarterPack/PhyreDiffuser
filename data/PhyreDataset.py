import cv2
import h5py
from torch.utils.data import Dataset
import phyre
import numpy as np


class PhyreDataset(Dataset):
    def __init__(self, file_path: str):
        """Base Dataset class for Phyre that more specific Dataset classes can inherit from

        Args:
            file_path (str): File to the HDF5 file with the dataset
            normalize (bool): Whether to normalize the images by calculating 2x / x.max - 1 so that the image is in range [-1,1]. This should technically be x-mean/std, but those are harder to calculate. Will maybe add this in the future.
            decode_image (bool): Set to true if images are raw returns from PHYRE simulations instead of already converted rgb images
            size (tuple): The size that an individual image will be resized to
        """
        self.file = h5py.File(file_path, "r")
        self.run_ids = []
        length = 0
        for template_id in self.file.keys():
            for task_id in self.file[template_id].keys():
                for run_id in self.file[template_id][task_id]:
                    length += 1
                    self.run_ids.append((template_id, task_id, run_id))
        self.length = length

    def __del__(self):
        try:
            self.file.close()
        except:
            pass

    def __len__(self):
        return self.length

    # features
    # x, y, cos_angle, sin_angle, diameter, color, shape
    def __getitem__(self, idx):
        (template_id, task_id, run_id) = self.run_ids[idx]
        x = np.array([self.preprocess_image(frame) for frame in self.file[template_id][task_id][run_id]])
        return x

    def preprocess_image(self, frame):
        return frame
