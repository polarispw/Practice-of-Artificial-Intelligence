import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset as BaseDataset


def generate_path_list(root="Image"):
    assert os.path.exists(root), f"File {root} does not exist"

    image_list = []
    label_list = []
    for i in os.listdir(root):
        path = os.path.join(root, i)
        temp = os.listdir(path)
        for name in temp:
            name = [os.path.join(path, name)]
            image_list += name

    print("{} images were found in the dataset.".format(len(image_list)))

    return image_list


class Dataset(BaseDataset):
    def __init__(self, images_list, augmentation=None, preprocessing=None):
        self.images_list = images_list
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        # values for 3 labels in gray picture
        self.class_values = [85, 170, 255]

    def __getitem__(self, i):
        # read data
        image = Image.open(self.images_list[i]).convert('RGB')
        image = np.array(image)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image)
            image = sample['image']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image)
            image = sample['image']

        return image

    def __len__(self):
        return len(self.images_list)


if __name__ == '__main__':

    valid_img_list = generate_path_list('Image')

    dataset = Dataset(valid_img_list)

    def visualize(**images):
        n = len(images)
        plt.figure(figsize=(16, 5))
        for i, (name, image) in enumerate(images.items()):
            plt.subplot(1, n, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.title(' '.join(name.split('_')).title())
            plt.imshow(image, cmap='gray')
        plt.show()

    image = dataset[0]
    visualize(image=image)
