import torch
from torch.utils.data import Dataset
import json
import os
import numpy as np 
import pandas as pd


class AnaxnetDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, filepath):
        """
        :param data_folder: folder where data files are stored
        :param transform: image transform pipeline
        """

        # Open word2vec embedding file
        self.rootdir = '/home/agun/mimic/dataset/VG/FeatureData/' 
        self.data = pd.read_csv(filepath, sep='\t')

        # Total number of datapoints
        self.dataset_size = len(self.data)


    def __getitem__(self, i):
        jsonFile = self.data['image_id'][i]
        filepath = os.path.join(self.rootdir, jsonFile)
        with open(filepath, 'r') as j:
            jsonData = json.load(j)
        imageID = jsonData['image_id']
        objects = jsonData['objects']
        image_features = torch.FloatTensor(np.array(jsonData['features']))
        target = torch.Tensor(np.array(jsonData['target']))
        #fix target being different
        if target.size()[0] != 18:
            target = torch.Tensor(np.zeros([18,9]))
        # print(target.size())
        return (imageID, objects, image_features), target

    def __len__(self):
        return self.dataset_size


