from __future__ import print_function
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

import sys
sys.path.append("/sailhome/tajwar/OOD-Detection/utils/")

# imports from our own scripts
from pytorch_pairwise_dataset import load_dataset
from pytorch_pairwise_dataset import check_dataset
from pytorch_pairwise_dataset import PairwiseDatasetRandom
from pytorch_pairwise_dataset import SelectiveClassDataset
from pytorch_pairwise_dataset import RandomlySampledDataset

from siamese_network import create_siamese_network_wrapper
from wide_resnet_pytorch import create_wide_resnet
from resnet_pytorch import ResNet18, ResNet34, ResNet50

from pytorch_pairwise_model_trainer import PairwiseModelTrainer

from plotting_log_utils import plot_loss
from plotting_log_utils import plot_accuracy
from get_readable_timestamp import get_readable_timestamp

def parse_arguments():
    ap = argparse.ArgumentParser()

    # dataset and model name and path arguments
    ap.add_argument("-dataset_name", "--dataset_name", type = str, default = "CIFAR10")
    ap.add_argument("-dataset_path", "--dataset_path", type = str)
    ap.add_argument("-shared_model_name", "--shared_model_name", type = str, default = "ResNet34")
    ap.add_argument("-version", "--version", type = int, default = 2, choices = [1, 2, 3, 4, 5])

    # training arguments
    ap.add_argument("-train_epoch_size", "--train_epoch_size", type = int, default = 10000)
    ap.add_argument("-train_batch_size", "--train_batch_size", type = int, default = 32)
    ap.add_argument("-val_batch_size", "--val_batch_size", type = int, default = 1024)
    ap.add_argument("-num_workers", "--num_workers", type = int, default = 4)
    ap.add_argument("-projection_dim", "--projection_dim", type = int, default = 128)

    ap.add_argument("-lr", "--lr", type = float, default = 0.01)
    ap.add_argument("-momentum", "--momentum", type = float, default = 0.9)
    ap.add_argument("-weight_decay", "--weight_decay", type = float, default = 0.0005)

    ap.add_argument("-num_epochs", "--num_epochs", type = int, default = 100)
    ap.add_argument("-use_nesterov", "--use_nesterov", type = int, default = 1)
    ap.add_argument("-verbose", "--verbose", type = bool, default = True)
    ap.add_argument("-use_default_scheduler", "--use_default_scheduler", type = int, default = 0, choices = [0, 1])
    ap.add_argument("-temp", "--temp", type = float, default = 0.1)

    # saving directory arguments
    ap.add_argument("-model_name", "--model_name", type = str, default = "CIFAR10")
    ap.add_argument("-model_save_path", "--model_save_path", type = str)
    ap.add_argument("-plot_directory", "--plot_directory", type = str, default = "./")
    ap.add_argument("-should_plot", "--should_plot", type = int, default = 0)
    ap.add_argument("-log_directory", "--log_directory", type = str, default = "./")

    # should we use partial dataset for training?
    ap.add_argument("-use_partial_dataset", "--use_partial_dataset", type = int, default = 0, choices = [0, 1])
    ap.add_argument("-partial_dataset_path_prefix", "--partial_dataset_path_prefix", type = str, default = "./")
    ap.add_argument("-partial_dataset_filename", "--partial_dataset_filename", type = str, default = "dataset_1/partial_dataset_labels.txt")

    # train on pairs formed of a smaller (sample) dataset taken from the original dataset
    ap.add_argument("-base_rate", "--base_rate", type = float, default = 1.0)

    # fine tune on a pretrained classifier
    ap.add_argument("-fine_tune", "--fine_tune", type = int, default = 0, choices = [0, 1, 2])
    ap.add_argument("-num_classes", "--num_classes", type = int, default = 10)
    ap.add_argument("-presaved_model_path", "--presaved_model_path", type = str, default = "./")

    script_arguments = vars(ap.parse_args())
    return script_arguments

# functions
def print_arguments(args):
    print()
    print("Arguments given for the script...")
    for key in args:
        print("Key: ", key, " Value: ", args[key])
    print()

def print_message(message):
    print()
    print(message)
    print()

def create_dataloaders(args):
    train_dataset = load_dataset(dataset_name = args["dataset_name"],
                                 dataset_path = args["dataset_path"],
                                 train = True,
                                 id_dataset_name = args["dataset_name"],
                                 id_dataset_path = args["dataset_path"],
                                 augment = True)

    if args["base_rate"] < 1.0:
        train_dataset = RandomlySampledDataset(dataset = train_dataset, base_rate = args["base_rate"], choose_randomly = False)

    validation_dataset = load_dataset(dataset_name = args["dataset_name"],
                                      dataset_path = args["dataset_path"],
                                      train = False,
                                      id_dataset_name = args["dataset_name"],
                                      id_dataset_path = args["dataset_path"],
                                      augment = False)

    if args["use_partial_dataset"] == 1:
        labels_path = args["partial_dataset_path_prefix"] + args["partial_dataset_filename"]
        train_dataset = SelectiveClassDataset(dataset = train_dataset, labels_path = labels_path)
        validation_dataset = SelectiveClassDataset(dataset = validation_dataset, labels_path = labels_path)

    check_dataset(dataset = train_dataset)
    check_dataset(dataset = validation_dataset)

    pairwise_train_dataset = PairwiseDatasetRandom(dataset = train_dataset, epoch_length = args["train_epoch_size"])
    pairwise_validation_dataset = PairwiseDatasetRandom(dataset = validation_dataset,
                                                        epoch_length = int(args["train_epoch_size"] / 5))

    # random pairs are formed during training, so shuffling is not necessary
    train_dataloader = torch.utils.data.DataLoader(dataset = pairwise_train_dataset,
                                                   batch_size = args["train_batch_size"],
                                                   shuffle = True,
                                                   num_workers = args["num_workers"])

    validation_dataloader = torch.utils.data.DataLoader(dataset = pairwise_validation_dataset,
                                                        batch_size = args["val_batch_size"],
                                                        shuffle = False,
                                                        num_workers = args["num_workers"])

    return train_dataloader, validation_dataloader

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
    architecture_map = process_shared_model_name_wide_resnet(shared_model_name = args["shared_model_name"])
    dataset_name = args["dataset_name"]

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
    dataset_name = args["dataset_name"]
    kwargs = {"model_name": args["shared_model_name"],
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

def create_classifier_model(args):
    if torch.cuda.is_available():
        device_type = "cuda"
    else:
        device_type = "cpu"

    device = torch.device(device_type)

    if args["shared_model_name"].find("WideResNet") == 0:
        model = create_wide_resnet_model(args)

    elif args["shared_model_name"].find("ResNet") >= 0:
        model = create_resnet_model(args)

    else:
        raise ValueError("Given model type is not supported")

    model.load_state_dict(torch.load(f = args["presaved_model_path"], map_location = device))
    print_message(message = "Model parameters loaded from saved information...")

    model.to(device)
    print_message(message = "Model loaded to appropriate device...")

    return model

def create_model(args):
    model = create_siamese_network_wrapper(shared_model_name = args["shared_model_name"],
                                           dataset_name = args["dataset_name"],
                                           load_to_gpu = torch.cuda.is_available(),
                                           version = args["version"],
                                           temp = args["temp"],
                                           projection_dim = args["projection_dim"],
                                           num_classes = args["num_classes"])

    if args["fine_tune"] > 0:
        classifier_model = create_classifier_model(args = args)
        model.shared_model = classifier_model

    return model

def create_optimizer_and_scheduler(args, model, len_train_dataloader):
    use_nesterov = None
    if args["use_nesterov"] == 1:
        use_nesterov = True
    elif args["use_nesterov"] == 0:
        use_nesterov = False
    else:
        raise ValueError("argument for using nesterov momentum, is not supported")

    fine_tune = args["fine_tune"]
    if fine_tune == 0 or fine_tune == 1:
        model_parameters = model.parameters()
    elif fine_tune == 2:
        model_parameters = model.last_layer.parameters()

    pairwise_optimizer = torch.optim.SGD(model_parameters,
                                         lr = args["lr"],
                                         momentum = args["momentum"],
                                         nesterov = use_nesterov,
                                         weight_decay = args["weight_decay"])

    if args["use_default_scheduler"] == 1:
        pairwise_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(pairwise_optimizer, T_max = args["num_epochs"])
    else:
        def cosine_annealing(step, total_steps, lr_max, lr_min):
            return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))

        # scheduler from outlier exposure
        pairwise_scheduler = torch.optim.lr_scheduler.LambdaLR(pairwise_optimizer,lr_lambda = lambda step: cosine_annealing(step,
                                                                                                                            args["num_epochs"] * len_train_dataloader,
                                                                                                                            1,  # since lr_lambda computes multiplicative factor
                                                                                                                            1e-6 / args["lr"]))

    return pairwise_optimizer, pairwise_scheduler

def run_model_training(args):
    model = create_model(args = args)
    if args["verbose"]:
        print("Model created.")
        print("Siamese network is based on: ", args["shared_model_name"])
        print(model)

    train_dataloader, val_dataloader = create_dataloaders(args = args)
    if args["verbose"]:
        print()
        print("Train dataloader and validation dataloader created")
        print()

    optimizer, scheduler = create_optimizer_and_scheduler(args = args, model = model, len_train_dataloader = len(train_dataloader))
    if args["verbose"]:
        print("Optimizer and scheduler created")
        print("Optimizer: ", optimizer)
        print("scheduler: ", scheduler)
        print("")

    readable_timestamp = get_readable_timestamp() + "_" + str(np.random.randint(10000))
    model_name = args["model_name"]
    model_path = os.path.join(args["model_save_path"], readable_timestamp)
    if not os.path.isdir(args["model_save_path"]):
        os.makedirs(args["model_save_path"])

    if args["verbose"]:
        print("Model name: ", model_name)
        print("model path: ", model_path)
        print("")

    model_trainer = PairwiseModelTrainer(model = model, model_name = model_name,
                                         train_loader = train_dataloader, validation_loader = val_dataloader,
                                         optimizer = optimizer, scheduler = scheduler)

    # should_use_scheduler -> True, use scheduler at every batch (instead of every epoch) like Outlier-exposure paper
    # should_use_scheduler -> False, use scheduler at every epoch
    if args["use_default_scheduler"] == 1:
        should_use_scheduler = False
    else:
        should_use_scheduler = True

    model_trainer.run_training(num_epochs = args["num_epochs"],
                               model_path = model_path,
                               verbose = args["verbose"],
                               should_use_scheduler = should_use_scheduler)

    model_trainer.report_peak_performance()

    plot_directory = os.path.join(args["plot_directory"], readable_timestamp)
    if not os.path.isdir(plot_directory):
        os.makedirs(plot_directory)

    if args["should_plot"] > 0:
        if args["verbose"]:
            print("Plot directory: ", plot_directory)
            print("Plotting loss and accuracy...")
            print()

        plot_loss(model_name, model_trainer.train_loss_history, model_trainer.val_loss_history, plot_directory)
        plot_accuracy(model_name, model_trainer.train_accuracy_history, model_trainer.val_accuracy_history, plot_directory)

    log_directory = os.path.join(args["log_directory"], readable_timestamp)
    if not os.path.isdir(log_directory):
        os.makedirs(log_directory)

    if args["verbose"]:
        print("Log directory: ", log_directory)
        print("Saving training log...")
        print()

    model_trainer.save_log(log_directory)

if __name__ == '__main__':
    script_arguments = parse_arguments()
    print_arguments(args = script_arguments)

    if torch.cuda.is_available():
        print()
        print("GPU available")
        print()

    run_model_training(args = script_arguments)
