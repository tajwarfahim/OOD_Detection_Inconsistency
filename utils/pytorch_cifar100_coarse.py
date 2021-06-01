# documentation help taken from:
# 1. https://github.com/ryanchankh/cifar100coarse/blob/master/sparse2coarse.py

# CIFAR-100 dataset with coarse (superclass) labels

from torch.utils.data import Dataset
from torchvision import datasets
import numpy as np

class CIFAR100Coarse(Dataset):
    def __init__(self, root, train, download, transform):
        self.dataset = datasets.CIFAR100(root = root, train = train, download = download, transform = transform)
        self.transform = transform
        
        self.coarse_labels = [4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
                              3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                              6, 11,  5, 10,  7,  6, 13, 15,  3, 15,
                              0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
                              5, 19,  8,  8, 15, 13, 14, 17, 18, 10,
                              16, 4, 17,  4,  2,  0, 17,  4, 18, 17,
                              10, 3,  2, 12, 12, 16, 12,  1,  9, 19,
                              2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
                              16, 19,  2,  4,  6, 19,  5,  5,  8, 19,
                              18,  1,  2, 15,  6,  0, 17,  8, 14, 13]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, fine_label = self.dataset[index]

        assert fine_label >= 0
        assert fine_label < len(self.coarse_labels)

        label = self.coarse_labels[fine_label]

        return img, label
