from torch.utils import data
from PIL import Image
import os
import pandas as pd
import numpy as np


class preprocess(data.Dataset):
    def __init__(self, path_to_dataset, path_to_annotations, transform_images=None):
        self.path_to_dataset = os.path.abspath(path_to_dataset)
        self.transform_images = transform_images

        self.listimg = []
        with open(self.path_to_dataset) as f:
            lines = f.readlines()
            for itr in lines:
                video = itr[-6:-1]
                pth = os.path.join(path_to_annotations, video+'.csv')
                df = pd.read_csv(pth)
                for entry in df.iterrows():
                    file = entry[1]['Video']
                    label = np.array(entry[1].iloc[1:-1])
                    file_path = os.path.join(itr.rstrip(), file)

                    # Choose files in which tools are present
                    if np.sum(label) == 0:
                        continue
                    if os.path.isfile(file_path):
                        self.listimg.append(file_path)

    def __len__(self):
        return len(self.listimg)

    def __getitem__(self, i):
        # Read the images and masks
        image = Image.open(os.path.join(self.listimg[i]))
        if self.transform_images is not None:
            image = self.transform_images(image)

        return image
