import numpy as np
from pathlib import Path
import cv2
from tqdm import tqdm

# Tiny images are combined vertically to 32x(32x64) images.
IMAGES_PATH = Path("/home/utku/Documents/repos/SSL_OOD/TinyImages")
HEIGHT = WIDTH = 32
CHANNELS = 3


def get_images_path_list(image_path):
    images_path_list = []
    for image_folder in image_path.iterdir():
        if image_folder.is_dir():
            for image in image_folder.iterdir():
                if image.suffix == ".jpg":
                    images_path_list.append(image)
    return images_path_list


def save_tiny_images_npz(image_path=IMAGES_PATH):
    images_path_list = get_images_path_list(image_path)
    X = np.ndarray(shape=(len(images_path_list), HEIGHT, WIDTH, CHANNELS), dtype=np.float32)
    Y = np.ndarray(shape=(len(images_path_list)), dtype=np.int32)
    for com_img_index, image_path in enumerate(images_path_list):
        combined_cv2_img = cv2.imread(str(image_path))
        # split the image in terms of columns
        for img_count in range(64):
            sing_img_index = com_img_index * 64 + img_count
            single_img = combined_cv2_img[img_count * 32 : (img_count + 1) * 32]
            X[sing_img_index] = single_img
            Y[sing_img_index] = 0
        del combined_cv2_img
    np.savez("tiny_images_dataset.npz", x=X, y=Y)


if __name__ == "__main__":
    save_tiny_images_npz()
