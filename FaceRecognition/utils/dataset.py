import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import numpy as np
import PIL


class FaceDataset(Dataset):
    def __init__(self, df_info: pd.DataFrame, transforms= None):
        super().__init__()
        self.dataframe = df_info
        self.transforms = transforms
        self.classes = sorted(self.dataframe["SUBJECT_ID"].unique())

        self.idx_to_class = {idx: cls for idx, cls in enumerate(self.classes)}
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.targets = np.array(
            [self.class_to_idx[cls] for cls in self.dataframe["SUBJECT_ID"].values]    
        )



    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):     
        image_row = self.dataframe.iloc[index]
        image = Image.open(image_row["PATH"])
        if self.transforms:
            image = self.transforms(image)
        image_label=self.class_to_idx[image_row['SUBJECT_ID']] 

        return image, image_label
