import cv2
from torch.utils.data import Dataset

class DeblurDataset(Dataset):
    def __init__(self, blur_paths, sharp_paths = None, transforms = None):
        self.X = blur_paths
        self.Y = sharp_paths
        self.transforms = transforms

    # number of items in the dataset
    def __len__(self):
        return(len(self.X))
    
    #Reads the i-th blurred image
    def __getitem__(self, i):
        blur_image = cv2.imread(f"./input/gaussian_blurred/{self.X[i]}")
        if self.transforms:
            blur_image = self.transforms(blur_image)
        if self.Y is not None:
            sharp_image = cv2.imread(f"./input/sharp/{self.Y[i]}")
            sharp_image = self.transforms(sharp_image)
            return (blur_image, sharp_image)
        else:
            return blur_image