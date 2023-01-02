from matplotlib import pyplot as plt
from random import random
import numpy as np
import torch


class RImages_300K(torch.utils.data.Dataset):
    def __init__(self, transform=None, data_path="/home/utku/Documents/repos/SSL_OOD/300K_random_images.npy"):
        dataset_np = np.load(data_path)
        dataset_np.shape
        self.dataset_np = np.load(data_path)
        self.offset = 0  # offset index
        self.transform = transform
        self.len = dataset_np.shape[0]

    def __getitem__(self, index):
        index = (index + self.offset) % self.__len__()
        img = self.dataset_np[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, 0  # 0 is the class -> Always OOD.

    def __len__(self):
        return self.len


if __name__ == "__main__" and "__file__" not in globals():
    temp_dataset = RImages_300K()

    def show_img():
        rand_int = int(random() * len(temp_dataset))
        print(rand_int)
        plt.imshow(temp_dataset[rand_int][0], interpolation="bicubic")
        plt.show()

    show_img()
