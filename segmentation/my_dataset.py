import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset as BaseDataset

class Dataset(BaseDataset):
    def __init__(
            self,
            images_dir,
            labels_dir,
            augmentation=None,
            preprocessing=None,
    ):
        self.idi = os.listdir(images_dir)
        self.idl = os.listdir(labels_dir)
        self.images_list = [os.path.join(images_dir, image_id) for image_id in self.idi]
        self.labels_list = [os.path.join(labels_dir, image_id) for image_id in self.idl]

        # values for 3 labels in gray picture
        self.class_values = [85, 170, 255]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = np.array(Image.open(self.images_list[i]))
        label = np.array(Image.open(self.labels_list[i]))
        # extract certain classes from mask (e.g. cars)
        masks = [(label == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.idi)

if __name__=='__main__':

    x_train_dir = "Heart Data/Image_DCM/png/Image/01"
    y_train_dir = "Heart Data/Image_DCM/png/Label/01"

    dataset = Dataset(x_train_dir, y_train_dir)

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

    image, mask = dataset[0]
    visualize(
        image=image,
        mask1=mask[:, :, 0],
        mask2=mask[:, :, 1],
        mask3=mask[:, :, 2],)