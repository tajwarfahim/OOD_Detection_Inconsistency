# wide resnet model names are in this format: WideResNet-depth-widen_factor-dropRate

# imports from general packages
import os
import torch
import numpy as np

# imports from our scripts
from .siamese_network_architectures import createSiameseNetwork
from .resnet_pytorch import ResNet18, ResNet34, ResNet50
from .wide_resnet_pytorch import create_wide_resnet

def print_message(message):
    print()
    print(message)
    print()

def create_resnet(model_name, num_classes, version, num_input_channels = 3):
    if version in [1, 2, 3, 4]:
        contains_last_layer = False
    elif version == 5:
        contains_last_layer = True
    else:
        raise ValueError("Given siamese network version is not supported.")

    if model_name == "ResNet18":
        resnet_model = ResNet18(contains_last_layer = contains_last_layer, num_input_channels = num_input_channels, num_classes = num_classes)

    elif model_name == "ResNet34":
        resnet_model = ResNet34(contains_last_layer = contains_last_layer, num_input_channels = num_input_channels, num_classes = num_classes)

    elif model_name == "ResNet50":
        resnet_model = ResNet50(contains_last_layer = contains_last_layer, num_input_channels = num_input_channels, num_classes = num_classes)

    else:
        raise ValueError('Model name not supported')

    return resnet_model

# process wide resnet model name in this format: WideResNet-depth-widen_factor-dropRate
def process_shared_model_name_wide_resnet(shared_model_name):
    assert shared_model_name.find("WideResNet") >= 0
    tokens = shared_model_name.split("-")
    assert len(tokens) == 4
    assert tokens[0] == "WideResNet"

    depth = int(tokens[1])
    widen_factor = int(tokens[2])
    dropRate = float(tokens[3])

    architecture_map = {"depth": depth, "widen_factor": widen_factor, "dropRate": dropRate}
    return architecture_map

def create_wide_resnet_based_shared_model(shared_model_name, dataset_name, num_classes, version = 1):
    shared_model = None
    architecture_map = process_shared_model_name_wide_resnet(shared_model_name = shared_model_name)

    if version in [1, 2, 3, 4]:
        contains_last_layer = False
    elif version == 5:
        contains_last_layer = True
    else:
        raise ValueError("Given siamese network version is not supported.")

    kwargs = {"depth": architecture_map["depth"],
              "widen_factor": architecture_map["widen_factor"],
              "dropRate": architecture_map["dropRate"],
              "num_classes": num_classes,
              "contains_last_layer": contains_last_layer}

    if dataset_name == "CIFAR10" or dataset_name == "CIFAR100" or dataset_name == "CIFAR" or dataset_name == "SVHN" or dataset_name == "CIFAR100Coarse":
        kwargs["num_input_channels"] = 3
        shared_model = create_wide_resnet(**kwargs)

    elif dataset_name == "MNIST":
        kwargs["num_input_channels"] = 1
        shared_model = create_wide_resnet(**kwargs)

    else:
        raise ValueError("Dataset name not supported")

    return shared_model

def create_resnet_based_shared_model(shared_model_name, dataset_name, num_classes, version = 1):
    shared_model = None

    if dataset_name == "CIFAR10" or dataset_name == "CIFAR100" or dataset_name == "CIFAR" or dataset_name == "SVHN" or dataset_name == "CIFAR100Coarse":
        shared_model = create_resnet(model_name = shared_model_name, num_input_channels = 3, num_classes = num_classes, version = version)

    elif dataset_name == "MNIST":
        shared_model = create_resnet(model_name = shared_model_name, num_input_channels = 1, num_classes = num_classes, version = version)

    else:
        raise ValueError("Dataset name not supported")

    return shared_model

def create_siamese_network_wrapper(dataset_name, shared_model_name, load_to_gpu, version = 1, temp = 0.1, projection_dim = 128, num_classes = 10):
    shared_model = None

    if shared_model_name.find("WideResNet") >= 0:
        shared_model = create_wide_resnet_based_shared_model(dataset_name = dataset_name,
                                                             shared_model_name = shared_model_name,
                                                             version = version,
                                                             num_classes = num_classes)

    elif shared_model_name.find("ResNet") >= 0:
        shared_model = create_resnet_based_shared_model(dataset_name = dataset_name,
                                                        shared_model_name = shared_model_name,
                                                        version = version,
                                                        num_classes = num_classes)

    else:
        raise ValueError("Model name seems improper")

    siamese_net = createSiameseNetwork(shared_model = shared_model, siamese_network_version = version, temp = temp, projection_dim = projection_dim)

    if load_to_gpu:
        if torch.cuda.is_available():
            siamese_net.cuda()
        else:
            raise ValueError("Model is asked to be loaded to gpu, but gpu is not available")

    return siamese_net
