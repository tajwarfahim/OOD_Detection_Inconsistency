from __future__ import print_function
import torch
import numpy as np
import time
import argparse
import os
import sys
import matplotlib.pyplot as plt
from torchvision import transforms as trn
from torchvision import datasets as dset


# imports from our scripts
from utils.msp_scorer import MSPScorer

from utils.pytorch_pairwise_dataset import load_dataset
from utils.pytorch_pairwise_dataset import RandomlySampledDataset
from utils.pytorch_pairwise_dataset import SelectiveClassDataset
from utils.pytorch_pairwise_dataset import get_mean_and_std_of_dataset
from utils.pytorch_pairwise_dataset import check_dataset

from utils.wide_resnet_pytorch import create_wide_resnet
from utils.resnet_pytorch import ResNet18, ResNet34, ResNet50

from utils.siamese_network import process_shared_model_name_wide_resnet

# creating arguments for the script
def parse_args():
    ap = argparse.ArgumentParser()

    # ID dataset
    ap.add_argument("-id_dataset_name", "--id_dataset_name", type = str, default = "CIFAR10")
    ap.add_argument("-num_classes", "--num_classes", type = int, default = 10)
    ap.add_argument("-id_dataset_path", "--id_dataset_path", type = str)
    ap.add_argument("-id_base_rate", "--id_base_rate", type = float, default = 1.0)

    # use first few id examples
    ap.add_argument("-use_first_samples", "--use_first_samples", type = int, default = 0, choices = [0, 1])
    ap.add_argument("-fixed_sample_size", "--fixed_sample_size", type = int, default = 20)

    # OOD dataset
    ap.add_argument("-ood_dataset_name", "--ood_dataset_name", type = str, default = "CIFAR100")
    ap.add_argument("-ood_dataset_path", "--ood_dataset_path", type = str)

    # Model info
    ap.add_argument("-model_type", "--model_type", type = str, default = "ResNet34")
    ap.add_argument("-presaved_model_path", "--presaved_model_path", type = str)
    ap.add_argument("-device_type", "--device_type", type = str, default = "cuda", choices = ["cpu", "cuda"])

    # Misc
    ap.add_argument("-num_workers", "--num_workers", type = int, default = 4)
    ap.add_argument("-batch_size", "--batch_size", type = int, default = 200)
    ap.add_argument("-verbose", "--verbose", type = bool, default = True)

    ap.add_argument("-id_selective_class", "--id_selective_class", type = int, default = 0, choices = [0, 1])
    ap.add_argument("-id_labels_path", "--id_labels_path", type = str, default = "./")
    ap.add_argument("-recall_level", "--recall_level", type = float, default = 0.95)

    script_arguments = vars(ap.parse_args())
    return script_arguments


def print_message(message):
    print()
    print(message)
    print()

def print_arguments(args):
    print()
    print("Arguments given for the script...")
    for key in args:
        print("Key: ", key, " Value: ", args[key])
    print()

def create_resnet_given_params(model_name, num_classes, num_input_channels):
    resnet_model = None

    if model_name == "ResNet18":
        resnet_model = ResNet18(contains_last_layer = True, num_input_channels = num_input_channels, num_classes = num_classes)

    elif model_name == "ResNet34":
        resnet_model = ResNet34(contains_last_layer = True, num_input_channels = num_input_channels, num_classes = num_classes)

    elif model_name == "ResNet50":
        resnet_model = ResNet50(contains_last_layer = True, num_input_channels = num_input_channels, num_classes = num_classes)

    else:
        raise ValueError('Model name not supported')

    return resnet_model

def create_wide_resnet_model(args):
    architecture_map = process_shared_model_name_wide_resnet(shared_model_name = args["model_type"])
    dataset_name = args["id_dataset_name"]

    kwargs = {"depth": architecture_map["depth"],
              "widen_factor": architecture_map["widen_factor"],
              "dropRate": architecture_map["dropRate"],
              "num_classes": args["num_classes"],
              "contains_last_layer": True}

    if dataset_name == "CIFAR10" or dataset_name == "CIFAR100" or dataset_name == "CIFAR" or dataset_name == "SVHN" or dataset_name == "CIFAR100Coarse":
        kwargs["num_input_channels"] = 3
        model = create_wide_resnet(**kwargs)

    elif dataset_name == "MNIST":
        kwargs["num_input_channels"] = 1
        model = create_wide_resnet(**kwargs)

    else:
        raise ValueError("Dataset name not supported")

    return model

def create_resnet_model(args):
    dataset_name = args["id_dataset_name"]
    kwargs = {"model_name": args["model_type"],
              "num_classes": args["num_classes"]}

    if dataset_name == "CIFAR10" or dataset_name == "CIFAR100" or dataset_name == "CIFAR" or dataset_name == "SVHN" or dataset_name == "CIFAR100Coarse":
        kwargs["num_input_channels"] = 3
        model = create_resnet_given_params(**kwargs)

    elif dataset_name == "MNIST":
        kwargs["num_input_channels"] = 1
        model = create_resnet_given_params(**kwargs)

    else:
        raise ValueError("Dataset name not supported")

    return model

def create_model(args):
    device_type = args["device_type"]
    if device_type not in ["cuda", "cpu"]:
        raise ValueError("Device type not supported")
    if device_type == "cuda":
        assert torch.cuda.is_available()
    device = torch.device(device_type)

    if args["model_type"].find("WideResNet") == 0:
        model = create_wide_resnet_model(args)

    elif args["model_type"].find("ResNet") >= 0:
        model = create_resnet_model(args)

    else:
        raise ValueError("Given model type is not supported")

    print_message(message = "Model architecture built...")

    print(model)

    model.load_state_dict(torch.load(f = args["presaved_model_path"], map_location = device))
    print_message(message = "Model parameters loaded from saved information...")

    model.to(device)
    print_message(message = "Model loaded to appropriate device...")

    return model


def create_dataloaders(args):
    id_train_dataset, mean = load_dataset(dataset_name = args["id_dataset_name"],
                                          dataset_path = args["id_dataset_path"],
                                          train = True,
                                          id_dataset_name = args["id_dataset_name"],
                                          id_dataset_path = args["id_dataset_path"],
                                          augment = False,
                                          return_mean = True)

    id_test_dataset = load_dataset(dataset_name = args["id_dataset_name"],
                                   dataset_path = args["id_dataset_path"],
                                   train = False,
                                   id_dataset_name = args["id_dataset_name"],
                                   id_dataset_path = args["id_dataset_path"],
                                   augment = False)

    # choose a certain per
    if args["use_first_samples"] > 0:
        assert args["fixed_sample_size"] > 0
        id_train_dataset = RandomlySampledDataset(dataset = id_train_dataset,
                                                  base_rate = 1.0,
                                                  choose_randomly = False,
                                                  fixed_sample_size = args["fixed_sample_size"])

    elif args["id_base_rate"] < 1.0:
        id_train_dataset = RandomlySampledDataset(dataset = id_train_dataset,
                                                  base_rate = args["id_base_rate"],
                                                  choose_randomly = False)

    ood_test_dataset = load_dataset(dataset_name = args["ood_dataset_name"],
                                    dataset_path = args["ood_dataset_path"],
                                    train = False,
                                    id_dataset_name = args["id_dataset_name"],
                                    id_dataset_path = args["id_dataset_path"],
                                    augment = False)


    if args["id_selective_class"] == 1:
        id_train_dataset = SelectiveClassDataset(dataset = id_train_dataset,
                                                 labels_path = args["id_labels_path"])

        id_test_dataset = SelectiveClassDataset(dataset = id_test_dataset,
                                                labels_path = args["id_labels_path"])

    check_dataset(dataset = id_train_dataset)
    check_dataset(dataset = id_test_dataset)

    id_train_dataloader = torch.utils.data.DataLoader(dataset = id_train_dataset,
                                                      batch_size = args["batch_size"],
                                                      shuffle = False,
                                                      num_workers = args["num_workers"],
                                                      pin_memory = True)

    id_test_dataloader = torch.utils.data.DataLoader(dataset = id_test_dataset,
                                                     batch_size = args["batch_size"],
                                                     shuffle = False,
                                                     num_workers = args["num_workers"],
                                                     pin_memory = True)

    ood_test_dataloader = torch.utils.data.DataLoader(dataset = ood_test_dataset,
                                                      batch_size = args["batch_size"],
                                                      shuffle = False,
                                                      num_workers = args["num_workers"],
                                                      pin_memory = True)

    return mean, id_train_dataloader, id_test_dataloader, ood_test_dataloader

def run_odin_experiment(args, scorer):
    # search grid for noise value used in mahalanobis paper's code
    noise_values = [0.0, 0.0005, 0.001, 0.0014, 0.002, 0.0024, 0.005, 0.01, 0.05, 0.1, 0.2]

    noise_values = list(set(noise_values))
    noise_values.sort()

    # search grid for temperature value used in mahalanobis paper's code
    temperature_values = [1.0, 10.0, 100.0, 1000.0]

    print("Running ODIN baseline.")
    print("Temperature values we try: ", temperature_values)
    print("Noise magnitude values we try: ", noise_values)

    best_auroc = None
    default_temperature = 1000.0
    default_noise = 0.0

    for temperature in temperature_values:
        for noise in noise_values:
            # we are in the MSP case, so we ignore this
            if temperature == 1.0 and noise == 0.0:
                continue

            scorer.calculate_odin_scores(temperature = temperature, noise = noise)
            auroc = scorer.calculate_auroc(score_type = "odin")
            aupr_in = scorer.calculate_aupr(score_type = "odin", positive = "in")
            aupr_out = scorer.calculate_aupr(score_type = "odin", positive = "out")
            fpr = scorer.calculate_fpr(score_type = "odin", recall_level = args["recall_level"])

            print()
            print("temperature: ", temperature, "noise: ", noise)
            print("AUROC: ", auroc)
            print("AUPR-In: ", aupr_in)
            print("AUPR-Out: ", aupr_out)
            print("FPR: ", fpr)
            print()

            if temperature == default_temperature and noise == default_noise:
                default_auroc = auroc
                default_aupr_in = aupr_in
                default_aupr_out = aupr_out
                default_fpr = fpr

            if best_auroc is None or auroc > best_auroc:
                best_noise_value = noise
                best_temperature_value = temperature
                best_auroc = auroc
                best_aupr_in = aupr_in
                best_aupr_out = aupr_out
                best_fpr = fpr

    print()
    print("Reporting ODIN baseline.")
    print("Noise value for best AUROC: ", best_noise_value)
    print("Temperature value for best AUROC: ", best_temperature_value)
    print("Best AUROC: ", best_auroc)
    print("Best AUPR-In: ", best_aupr_in)
    print("Best AUPR-Out: ", best_aupr_out)
    print("Best FPR: ", best_fpr)
    print()

    print()
    print("Default temperature: ", default_temperature, "Default noise: ", default_noise)
    print("Default AUROC: ", default_auroc)
    print("Default AUPR-In: ", default_aupr_in)
    print("Default AUPR-Out: ", default_aupr_out)
    print("Default FPR: ", default_fpr)
    print()

def run_mahalanobis_experiment(scorer, args):
    noise_values = [0, 0.0005, 0.001, 0.0014, 0.002, 0.0024, 0.005, 0.01, 0.05, 0.1, 0.2]

    best_auroc = None
    default_noise = 0.0

    for noise in noise_values:
        scorer.calculate_mahalanobis_scores(noise = noise)

        auroc = scorer.calculate_auroc(score_type = "mahalanobis")
        aupr_in = scorer.calculate_aupr(score_type = "mahalanobis", positive = "in")
        aupr_out = scorer.calculate_aupr(score_type = "mahalanobis", positive = "out")
        fpr = scorer.calculate_fpr(score_type = "mahalanobis", recall_level = args["recall_level"])

        print()
        print("noise: ", noise)
        print("AUROC: ", auroc)
        print("AUPR-In: ", aupr_in)
        print("AUPR-Out: ", aupr_out)
        print("FPR: ", fpr)
        print()

        if noise == default_noise:
            default_auroc = auroc
            default_aupr_in = aupr_in
            default_aupr_out = aupr_out
            default_fpr = fpr

        if best_auroc is None or auroc > best_auroc:
            best_noise_value = noise
            best_auroc = auroc
            best_aupr_in = aupr_in
            best_aupr_out = aupr_out
            best_fpr = fpr

    print()
    print("Reporting Mahalanobis baseline.")
    print("Noise value for best AUROC: ", best_noise_value)
    print("Best AUROC: ", best_auroc)
    print("Best AUPR-In: ", best_aupr_in)
    print("Best AUPR-Out: ", best_aupr_out)
    print("Best FPR: ", best_fpr)
    print()

    print()
    print("Default noise: ", default_noise)
    print("Default AUROC: ", default_auroc)
    print("Default AUPR-In: ", default_aupr_in)
    print("Default AUPR-Out: ", default_aupr_out)
    print("Default FPR: ", default_fpr)
    print()


def run_script(args):
    start_time = time.time()

    model = create_model(args = args)

    mean, id_train_dataloader, id_test_dataloader, ood_test_dataloader = create_dataloaders(args = args)
    print()
    print("Data loaders are created.")
    print("ID train dataset mean: ", mean)
    print()

    scorer = MSPScorer(model = model,
                       id_train_dataloader = id_train_dataloader,
                       id_test_dataloader = id_test_dataloader,
                       ood_test_dataloader = ood_test_dataloader,
                       num_classes = args["num_classes"],
                       id_train_dataset_mean = mean)

    accuracy = scorer.calculate_accuracy()

    print()
    print("Model's accuracy on ID test dataset: ", accuracy)
    print()

    scorer.calculate_msp_scores()
    msp_auroc = scorer.calculate_auroc(score_type = "msp")
    msp_aupr_in = scorer.calculate_aupr(score_type = "msp", positive = "in")
    msp_aupr_out = scorer.calculate_aupr(score_type = "msp", positive = "out")
    msp_fpr = scorer.calculate_fpr(score_type = "msp", recall_level = args["recall_level"])

    print()
    print("MSP AUROC: ", msp_auroc)
    print("MSP AUPR-In: ", msp_aupr_in)
    print("MSP AUPR-Out: ", msp_aupr_out)
    print("MSP FPR: ", msp_fpr)
    print()

    run_odin_experiment(args = args, scorer = scorer)
    run_mahalanobis_experiment(scorer = scorer, args = args)

    end_time = time.time()

    total_time = end_time - start_time
    print()
    print("Total time to run this experiment: ", total_time, " s")
    print()

if __name__ == '__main__':
    script_arguments = parse_args()
    print_arguments(args = script_arguments)
    run_script(args = script_arguments)
