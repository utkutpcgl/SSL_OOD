import numpy as np
import torch
from bisect import bisect_left
from pathlib import Path
from PIL import Image


# TODO read pil images and get only the desired slices for a particular image. But it will be slow. Split the 64 images in my opinion. But optimization can come later.
class TinyImages(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        dataset_folder = Path("/home/utku/Documents/repos/SSL_OOD/TinyImages")
        num_of_images_per_image = 64
        image_path_list = self.get_images_path_list(dataset_folder)
        self.len = len(image_path_list) * num_of_images_per_image
        image_path_list.sort()
        image_width = 32
        image_height = 32

        def load_image(idx):
            """There 64 images in one image."""
            container_img_index = idx // num_of_images_per_image
            contained_img_path = image_path_list[container_img_index]
            img_row_idx = idx % num_of_images_per_image
            pil_img = Image.open(contained_img_path)
            left, top, right, bottom = 0, img_row_idx * image_height, image_width, (img_row_idx + 1) * image_height
            pil_img_crop = pil_img.crop((left, top, right, bottom))
            return pil_img_crop

        self.load_image = load_image
        self.transform = transform

    def __getitem__(self, index):
        # This is necessary due to strange offestting in the code.
        index = index % self.len
        img = self.load_image(index)
        if self.transform is not None:
            img = self.transform(img)

        return img, 0  # 0 is the class

    def __len__(self):
        return self.len

    @staticmethod
    def get_images_path_list(image_path):
        images_path_list = []
        for image_folder in image_path.iterdir():
            if image_folder.is_dir():
                for image in image_folder.iterdir():
                    if image.suffix == ".jpg":
                        images_path_list.append(image)
        return images_path_list
