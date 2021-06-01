# datasets here return x1, x2, label
# label -> 0 if x1 and x2 are from the same class
# label -> 1 if x1 and x2 are from different classes

# imports from general packages
import os
import torch
import numpy as np
from collections import defaultdict
from torchvision import datasets
from torch.utils.data import Dataset
from torchvision import transforms

# imports from our scripts
from .pytorch_cifar100_coarse import CIFAR100Coarse

# Citation:
# 1. https://discuss.pytorch.org/t/about-normalization-using-pre-trained-vgg16-networks/23560/6
def get_mean_and_std_of_dataset(dataset):
    single_img, _ = dataset[0]
    assert torch.is_tensor(single_img)
    num_channels, dim_1, dim_2 = single_img.shape[0], single_img.shape[1], single_img.shape[2]

    loader = torch.utils.data.DataLoader(dataset, batch_size = 128, num_workers = 4, shuffle = False)
    mean = 0.0
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
    mean = mean / len(loader.dataset)

    var = 0.0
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        var += ((images - mean.unsqueeze(1)) ** 2).sum([0,2])

    std = torch.sqrt(var / (len(loader.dataset) * dim_1 * dim_2))

    return mean, std

def print_dataset_information(dataset, dataset_name, train, verbose):
    if verbose:
        print()
        print("Dataset name:", dataset_name)
        print("Is train: ", train)
        print("Number of elements: ", len(dataset))

        img, _ = dataset[0]
        print(img.shape)
        print()

        print("Transform: ", dataset.transform)
        print()

def load_dataset_with_basic_transform(dataset_name, train, path):
    transform = transforms.ToTensor()

    if dataset_name == "CIFAR10":
        dataset = datasets.CIFAR10(root = path, train = train, download = False, transform = transform)

    elif dataset_name == "CIFAR100" or dataset_name == "CIFAR100Coarse":
        dataset = datasets.CIFAR100(root = path, train = train, download = False, transform = transform)

    elif dataset_name == "SVHN":
        split = "test"
        if train:
            split = "train"

        dataset = datasets.SVHN(root = path, split = split, download = False, transform = transform)

    elif dataset_name == "MNIST":
        dataset = datasets.MNIST(root = path, train = train, download = False, transform = transform)

    else:
        raise ValueError("Given dataset name is not supported.")

    return dataset

def get_custom_data_transform(dataset_name, augment, mean, std):
    if dataset_name == "SVHN":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    elif augment:
        if dataset_name in ["CIFAR10", "CIFAR100", "CIFAR100Coarse", "MNIST"]:
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

        else:
            raise ValueError("Given dataset name is not supported.")

    else:
        if dataset_name in ["CIFAR10", "CIFAR100", "CIFAR100Coarse", "MNIST"]:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

        elif dataset_name in ["CelebA", "STL10", "ImageFolder"]:
            transform = transforms.Compose([
                transforms.Resize(size = (32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

        else:
            raise ValueError("Given dataset name is not supported.")

    return transform

def load_dataset_with_custom_data_transform(dataset_name, train, path, transform):
    if dataset_name == "CIFAR10":
        dataset = datasets.CIFAR10(root = path, train = train, download = False, transform = transform)

    elif dataset_name == "CIFAR100":
        dataset = datasets.CIFAR100(root = path, train = train, download = False, transform = transform)

    elif dataset_name == "CIFAR100Coarse":
        dataset = CIFAR100Coarse(root = path, train = train, download = False, transform = transform)

    elif dataset_name == "MNIST":
        dataset = datasets.MNIST(root = path, train = train, download = False, transform = transform)

    elif dataset_name == "CelebA":
        split = "test"
        if train:
            split = "train"

        dataset = datasets.CelebA(root = path, split = split, download = False, transform = transform)

    elif dataset_name == "SVHN":
        split = "test"
        if train:
            split = "train"

        dataset = datasets.SVHN(root = path, split = split, download = False, transform = transform)

    elif dataset_name == "STL10":
        split = "test"
        if train:
            split = "train"

        dataset = datasets.STL10(root = path, split = split, download = False, transform = transform)

    elif dataset_name == "ImageFolder":
        dataset = datasets.ImageFolder(root = path, transform = transform)

    else:
        raise ValueError("Dataset is not supported")

    return dataset

def load_dataset(dataset_name, dataset_path, train, id_dataset_name, id_dataset_path, augment, return_mean = False):
    id_train_dataset = load_dataset_with_basic_transform(dataset_name = id_dataset_name, train = True, path = id_dataset_path)
    mean, std = get_mean_and_std_of_dataset(dataset = id_train_dataset)

    print()
    print("Mean: ", mean)
    print("Std: ", std)
    print()

    custom_transform = get_custom_data_transform(dataset_name = dataset_name, augment = augment, mean = mean, std = std)
    dataset = load_dataset_with_custom_data_transform(dataset_name = dataset_name, train = train, path = dataset_path, transform = custom_transform)

    print()
    print("###########################")
    print("printing information on the loaded dataset!")
    print("###########################")
    print()

    print_dataset_information(dataset = dataset, dataset_name = dataset_name, train = train, verbose = True)

    if return_mean:
        return dataset, mean
    else:
        return dataset

def create_label_to_index_mapping(dataset):
    mapping = defaultdict(list)
    for index in range(len(dataset)):
        _, label = dataset[index]
        if torch.is_tensor(label):
            label = label.item()
        mapping[label].append(index)

    return mapping

def generate_per_class_sample(dataset_name, dataset_path, save_path, train, sample_size):
    dataset = load_dataset(dataset_name = dataset_name, train = train, path = dataset_path, download = True)
    mapping = create_label_to_index_mapping(dataset)
    labels = [key for key in mapping.keys()]
    labels.sort()
    print("Labels: ", labels)

    per_class_samples = []

    for label in labels:
        indices = np.random.choice(a = mapping[label], size = sample_size, replace = False)
        indices = np.sort(a = indices)

        sample = [label] + [indices[i] for i in range(indices.shape[0])]
        per_class_samples.append(sample)

    np.savetxt(save_path, np.array(per_class_samples, dtype = np.int32), delimiter=",", fmt="%d")

def load_per_class_samples(save_path):
    assert os.path.isfile(save_path)
    class_to_sample_indices_mapping = {}

    with open(save_path, 'r') as file:
        for line in file:
            tokens = line.split(",")
            label = int(tokens[0])
            samples = [int(tokens[i]) for i in range(1, len(tokens))]

            class_to_sample_indices_mapping[label] = samples

    return class_to_sample_indices_mapping

# prepare validation dataset
def generate_validation_dataset(dataset_name, dataset_path, save_path, half_validation_dataset_size):
    dataset = load_dataset(dataset_name = dataset_name, train = False, path = dataset_path, download = True)
    mapping = create_label_to_index_mapping(dataset)
    labels = [key for key in mapping.keys()]
    labels.sort()
    print("Labels: ", labels)

    validation_dataset = []

    for i in range(half_validation_dataset_size * 2):
        # same class -> label 0
        if i % 2 == 0:
            random_label = np.random.choice(labels)
            indices = np.random.choice(a = mapping[random_label], size = 2, replace = False)
            label = 0
            data_point = [indices[0], indices[1], label]

        # different class -> label 1
        else:
            random_labels = np.random.choice(a = labels, size = 2, replace = False)
            index_1 = np.random.choice(a = mapping[random_labels[0]])
            index_2 = np.random.choice(a = mapping[random_labels[1]])
            label = 1
            data_point = [index_1, index_2, label]

        validation_dataset.append(data_point)

    np.savetxt(save_path, np.array(validation_dataset, dtype= np.int32), delimiter=",", fmt = "%d")

def load_validation_dataset(save_path):
    assert os.path.isfile(save_path)

    validation_dataset = []
    with open(save_path, 'r') as file:
        for line in file:
            tokens = line.split(",")
            assert len(tokens) == 3
            data_point = [int(tokens[0]), int(tokens[1]), int(tokens[2])]

            validation_dataset.append(data_point)

    return validation_dataset

def is_valid_class_to_sample_mapping(mapping):
    mapped_list_size = -1
    for key in mapping:
        mapped_list = mapping[key]
        if mapped_list_size == -1:
            mapped_list_size = len(mapped_list)
        elif mapped_list_size != len(mapped_list):
            return False

    return True

class PreSavedPerClassSampledDataset(Dataset):
    def __init__(self, dataset, sample_indices_path):
        self.dataset = dataset

        self.class_to_sample_indices_mapping = load_per_class_samples(save_path = sample_indices_path)
        assert is_valid_class_to_sample_mapping(self.class_to_sample_indices_mapping)

        self.class_labels = [class_label for class_label in self.class_to_sample_indices_mapping]
        assert len(self.class_labels) > 0
        self.class_labels.sort()
        self.sample_size_per_class = len(self.class_to_sample_indices_mapping[self.class_labels[0]])

        img_indices = []
        for class_label in self.class_labels:
            img_indices = img_indices + self.class_to_sample_indices_mapping[class_label]
        self.img_indices = img_indices

        self.len = len(self.img_indices)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        img_index = self.img_indices[index]
        return self.dataset[img_index]

    def get_class_labels(self):
        return self.class_labels

    def get_sample_size_per_class(self):
        return self.sample_size_per_class

    def get_sample_from_class(self, class_label, sample_index):
        assert sample_index >= 0 and sample_index < self.sample_size_per_class

        sample_datapoint_indices = self.class_to_sample_indices_mapping[class_label]
        img_index = sample_datapoint_indices[sample_index]

        return self.dataset[img_index]

# pairwise dataset, with random pairings made over the given dataset
# label = 0, both data points from the pair are from the same class
# label = 1, datapoints from the pair are from different classes
class PairwiseDatasetRandom(Dataset):
    def __init__(self, dataset, epoch_length = 10000):
        self.dataset = dataset
        self.mapping = create_label_to_index_mapping(self.dataset)
        self.labels = [key for key in self.mapping.keys()]
        self.len = epoch_length

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        img1 = None
        img2 = None
        label = None

        # same class -> label = 0
        if index % 2 == 0:
            random_label = np.random.choice(self.labels)
            indices = np.random.choice(a = self.mapping[random_label], size = 2, replace = False)
            img1, _ = self.dataset[indices[0]]
            img2, _ = self.dataset[indices[1]]

            label = 0

        # different class -> label = 1
        else:
            random_labels = np.random.choice(a = self.labels, size = 2, replace = False)
            index_1 = np.random.choice(a = self.mapping[random_labels[0]])
            index_2 = np.random.choice(a = self.mapping[random_labels[1]])
            img1, _ = self.dataset[index_1]
            img2, _ = self.dataset[index_2]

            label = 1

        return img1, img2, label

# pairwise validation dataset
class PairwiseDatasetPreSaved(Dataset):
    def __init__(self, dataset, combination_path):
        self.dataset = dataset
        self.combinations = load_validation_dataset(save_path = combination_path)
        self.len = len(self.combinations)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        data_point = self.combinations[index]
        index_1, index_2, label = data_point[0], data_point[1], data_point[2]

        img1, _ = self.dataset[index_1]
        img2, _ = self.dataset[index_2]

        return img1, img2, label


# dataset for classification task (not pairwise task)
# that contains selected classes
class SelectiveClassDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, labels_path):
        self.dataset = dataset

        self.partial_labels = []
        f = open(labels_path, 'r')
        for line in f:
            self.partial_labels.append(int(line))
        f.close()

        self.subset_indices = []
        for index in range(len(self.dataset)):
            _, label = self.dataset[index]
            if label in self.partial_labels:
                self.subset_indices.append(index)

        self.len = len(self.subset_indices)

        self.remapped_labels = {}
        for i in range(len(self.partial_labels)):
            label = self.partial_labels[i]
            self.remapped_labels[label] = i

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        img, label = self.dataset[self.subset_indices[index]]
        if torch.is_tensor(label):
            label = label.item()

        new_label = self.remapped_labels[label]
        return img, new_label

# dataset that contains randomly sampled datapoints of the original dataset
class RandomlySampledDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, base_rate, choose_randomly = True, fixed_sample_size = None):
        self.dataset = dataset

        assert base_rate > 0.0 and base_rate <= 1.0
        self.base_rate = base_rate

        label_to_index_mapping = create_label_to_index_mapping(dataset = self.dataset)
        labels = [int(key) for key in label_to_index_mapping.keys()]
        labels.sort()

        print("Labels: ", labels)

        sample_indices = []
        for label in labels:
            indices = label_to_index_mapping[label]

            if fixed_sample_size is not None:
                sample_size = fixed_sample_size
            else:
                sample_size = int(len(indices) * self.base_rate)

            if choose_randomly:
                class_sample = np.random.choice(a = indices, size = sample_size, replace = False)
                class_sample = class_sample.tolist()
            else:
                class_sample = indices[0 : sample_size]

            sample_indices = sample_indices + class_sample

        self.sample_indices = sample_indices
        self.len = len(self.sample_indices)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        new_index = self.sample_indices[index]
        return self.dataset[new_index]

def check_dataset(dataset):
    label_to_index_mapping = create_label_to_index_mapping(dataset = dataset)
    labels = [key for key in label_to_index_mapping.keys()]
    labels.sort()

    print()
    print("Printing classes and number of element in each class.")
    print()

    for label in labels:
        print("Class: ", label, " Num datapoints: ", len(label_to_index_mapping[label]))

    print()
