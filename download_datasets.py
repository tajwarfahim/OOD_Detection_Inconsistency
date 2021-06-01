import argparse
from torchvision import datasets
from torchvision import transforms
import os

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-dataset_path_prefix", "--dataset_path_prefix", type = str)
    ap.add_argument("-dataset_name", "--dataset_name", type = str, default = "all", choices = ["CIFAR10", "CIFAR100", "SVHN", "Omniglot", "CelebA", "STL10", "all"])
    script_arguments = vars(ap.parse_args())

    return script_arguments

def print_dataset_information(dataset, dataset_name, train):
    print()
    print("Dataset name:", dataset_name)
    print("Is train: ", train)
    print("Number of elements: ", len(dataset))

    img, _ = dataset[0]
    print(img.shape)
    print()

def download_individual_dataset(dataset_name, path, train):
    transform = transforms.Compose([transforms.ToTensor()])

    if dataset_name == "CIFAR10":
        dataset = datasets.CIFAR10(root = path, train = train, download = True, transform = transform)

    elif dataset_name == "CIFAR100":
        dataset = datasets.CIFAR100(root = path, train = train, download = True, transform = transform)

    elif dataset_name == "MNIST":
        dataset = datasets.MNIST(root = path, train = train, download = True, transform = transform)

    elif dataset_name == "CelebA":
        if train:
            split = "train"
        else:
            split = "test"

        dataset = datasets.CelebA(root = path, split = split, download = True, transform = transform)

    elif dataset_name == "SVHN":
        if train:
            split = "train"
        else:
            split = "test"

        dataset = datasets.SVHN(root = path, split = split, download = True, transform = transform)

    elif dataset_name == "Omniglot":
        if train:
            background = True
        else:
            background = False

        dataset = datasets.Omniglot(root = path, background = background, download = True, transform = transform)

    elif dataset_name == "STL10":
        if train:
            split = "train"
        else:
            split = "test"

        dataset = datasets.STL10(root = path, split = split, download = True, transform = transform)

    else:
        raise ValueError("Dataset is not supported")

    print_dataset_information(dataset = dataset, dataset_name = dataset_name, train = train)

def download_datasets(args):
    path_prefix = args["dataset_path_prefix"]

    if args["dataset_name"] == "all":
        dataset_names = ["CIFAR10", "CIFAR100", "SVHN", "Omniglot", "STL10", "CelebA"]
    else:
        dataset_names = [args["dataset_name"]]

    for dataset_name in dataset_names:
        print()
        print("Downloading dataset: ", dataset_name)

        download_path = os.path.join(path_prefix, dataset_name)
        print("Downloading to this directory: ", download_path)
        print()

        download_individual_dataset(dataset_name = dataset_name, path = download_path, train = True)
        download_individual_dataset(dataset_name = dataset_name, path = download_path, train = False)

if __name__ == '__main__':
    script_arguments = parse_args()
    download_datasets(args = script_arguments)
