import torchvision.utils

import config
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import csv
import matplotlib.pyplot as plt


class DRDataset(Dataset):
    def __init__(self, images_folder, path_to_csv, train=True, transform=None):
        super().__init__()
        self.data = pd.read_csv(path_to_csv)
        self.images_folder = images_folder
        self.image_files = os.listdir(images_folder)
        self.transform = transform
        self.train = train

        label_dict = {
            "Image name": [],
            "Retinopathy grade": [],
            # "Risk of macular edema ": []
        }
        with open(path_to_csv, 'r') as file:
            data_DictReader = csv.DictReader(file)
            for row in data_DictReader:
                label_dict["Image name"].append(row["Image name"])
                label_dict["Retinopathy grade"].append(row["Retinopathy grade"])
                # label_dict["Risk of macular edema "].append(row["Risk of macular edema "])

        assert len(label_dict["Image name"]) == len(label_dict["Retinopathy grade"])
        num_label = len(label_dict["Image name"])

        self.filenames = []
        self.labels = []
        for i in range(num_label):
            self.filenames.append(label_dict["Image name"][i])
            self.labels.append(0 if label_dict["Retinopathy grade"][i] <= '1' else 1)
            # self.labels[2].append(label_dict["Risk of macular edema "][i])

    def __len__(self):
        return self.data.shape[0] if self.train else len(self.image_files)

    def __getitem__(self, index):
        # if self.train:
        #     image_file, label = self.filenames[index], self.labels[index]
        # else:
        #     # if test simply return -1 for label, I do this in order to
        #     # re-use same dataset class for test set submission later on
        #     image_file, label = self.image_files[index], -1
        #     image_file = image_file.replace(".jpg", "")

        image_file, label = self.filenames[index], self.labels[index]
        image = np.array(Image.open(os.path.join(self.images_folder, image_file+".jpg")))

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, label, image_file


if __name__ == "__main__":
    """
    Test if everything works ok
    """
    dataset = DRDataset(
        images_folder="./train_images_resized_512/",
        path_to_csv="./labels/train.csv",
        transform=config.val_transforms,
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=8,
        #num_workers=2,
        shuffle=True,
        pin_memory=True
    )

    for step, (image, label, file) in tqdm(enumerate(loader)):
        if step <= 2:
            print('step:%a' %step)
            print(image.shape)
            print(image.shape)

            for index,im in enumerate(image):
                plt.subplot(2,int(image.shape[0]/2), index+1)
                plt.imshow(im.permute(1,2,0))
            plt.show()

    import sys
    sys.exit()