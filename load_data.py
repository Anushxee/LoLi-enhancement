import os
import cv2

class LoLIDataset:
    def __init__(self, low_dir, gt_dir):
        self.low_dir = low_dir
        self.gt_dir = gt_dir

        self.low_images = sorted(os.listdir(low_dir))
        self.gt_images = sorted(os.listdir(gt_dir))

    def __len__(self):
        return len(self.low_images)

    def get_pair(self, idx):
        low_path = os.path.join(self.low_dir, self.low_images[idx])
        gt_path = os.path.join(self.gt_dir, self.gt_images[idx])

        low_img = cv2.imread(low_path)
        gt_img = cv2.imread(gt_path)

        return low_img, gt_img, self.low_images[idx]