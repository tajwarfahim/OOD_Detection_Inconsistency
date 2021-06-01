from __future__ import print_function
import torch
import numpy as np
import argparse
import os
import sys

# import from our scripts
from utils.pytorch_pairwise_dataset import load_dataset
from utils.pytorch_pairwise_dataset import RandomlySampledDataset
from utils.pytorch_pairwise_dataset import SelectiveClassDataset
from utils.pytorch_pairwise_dataset import check_dataset

from utils.wide_resnet_pytorch import create_wide_resnet
from utils.resnet_pytorch import ResNet18, ResNet34, ResNet50

from utils.pytorch_classifier_trainer import ClassifierTrainer

from utils.siamese_network import process_shared_model_name_wide_resnet

from utils.plotting_log_utils import plot_loss
from utils.plotting_log_utils import plot_accuracy

from utils.get_readable_timestamp import get_readable_timestamp

def parse_arguments():
    ap = argparse.ArgumentParser()

    # dataset and model name and path arguments
    ap.add_argument("-dataset_name", "--dataset_name", type = str, default = "CIFAR10")
    ap.add_argument("-dataset_path", "--dataset_path", type = str)
    ap.add_argument("-num_classes", "--num_classes", type = int, default = 10)
    ap.add_argument("-model_type", "--model_type", type = str, default = "ResNet34")

    # training arguments
    ap.add_argument("-train_batch_size", "--train_batch_size", type = int, default = 32)
    ap.add_argument("-val_batch_size", "--val_batch_size", type = int, default = 1024)
    ap.add_argument("-num_workers", "--num_workers", type = int, default = 4)

    ap.add_argument("-lr", "--lr", type = float, default = 0.1)
    ap.add_argument("-momentum", "--momentum", type = float, default = 0.9)
    ap.add_argument("-weight_decay", "--weight_decay", type = float, default = 0.0005)

    ap.add_argument("-num_epochs", "--num_epochs", type = int, default = 200)
    ap.add_argument("-use_nesterov", "--use_nesterov", type = int, default = 1)
    ap.add_argument("-verbose", "--verbose", type = bool, default = True)
    ap.add_argument("-use_default_scheduler", "--use_default_scheduler", type = int, default = 0, choices = [0, 1])
    ap.add_argument("-should_plot", "--should_plot", type = int, default = 0)

    # saving directory arguments
    ap.add_argument("-model_name", "--model_name", type = str, default = "MSP model")
    ap.add_argument("-model_save_path", "--model_save_path", type = str)
    ap.add_argument("-plot_directory", "--plot_directory", type = str, default = "./")
    ap.add_argument("-log_directory", "--log_directory", type = str, default = "./")

    # training on partial datasets
    ap.add_argument("-base_rate", "--base_rate", type = float, default = 1.0)
    ap.add_argument("-use_partial_dataset", "--use_partial_dataset", type = int, default = 0, choices = [0, 1])
    ap.add_argument("-partial_dataset_path_prefix", "--partial_dataset_path_prefix", type = str, default = "./")
    ap.add_argument("-partial_dataset_filename", "--partial_dataset_filename", type = str, default = "dataset_1/partial_dataset_labels.txt")

    script_arguments = vars(ap.parse_args())
    return script_arguments

def print_arguments(args):
    print()
    print("Arguments given for the script...")
    for key in args:
        print("Key: ", key, " Value: ", args[key])
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

    train_dataloader = torch.utils.data.DataLoader(dataset = train_dataset,
                                                   batch_size = args["train_batch_size"],
                                                   shuffle = True,
                                                   num_workers = args["num_workers"])

    validation_dataloader = torch.utils.data.DataLoader(dataset = validation_dataset,
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
    architecture_map = process_shared_model_name_wide_resnet(shared_model_name = args["model_type"])
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
    if args["model_type"].find("WideResNet") == 0:
        model = create_wide_resnet_model(args)

    elif args["model_type"].find("ResNet") >= 0:
        model = create_resnet_model(args)

    else:
        raise ValueError("Given model type is not supported")

    if torch.cuda.is_available():
        model.cuda()

    return model

def create_optimizer_and_scheduler(args, model, len_train_dataloader):
    use_nesterov = None
    if args["use_nesterov"] == 1:
        use_nesterov = True
    elif args["use_nesterov"] == 0:
        use_nesterov = False
    else:
        raise ValueError("argument for using nesterov momentum, is not supported")

    optimizer = torch.optim.SGD(model.parameters(),
                                lr = args["lr"],
                                momentum = args["momentum"],
                                nesterov = use_nesterov,
                                weight_decay = args["weight_decay"])

    if args["use_default_scheduler"] == 1:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = args["num_epochs"])
        
    else:
        def cosine_annealing(step, total_steps, lr_max, lr_min):
            return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))

        # scheduler from outlier exposure
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda = lambda step: cosine_annealing(step,
                                                                                                          args["num_epochs"] * len_train_dataloader,
                                                                                                          1,  # since lr_lambda computes multiplicative factor
                                                                                                          1e-6 / args["lr"]))

    return optimizer, scheduler

def run_model_training(args):
    model = create_model(args = args)
    if args["verbose"]:
        print("Model created.")
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

    readable_timestamp = get_readable_timestamp() + "_" + str(np.random.randint(1000000))
    model_name = args["model_name"]
    model_path = os.path.join(args["model_save_path"], readable_timestamp)
    if not os.path.isdir(args["model_save_path"]):
        os.makedirs(args["model_save_path"])

    if args["verbose"]:
        print("Model name: ", model_name)
        print("model path: ", model_path)
        print("")

    model_trainer = ClassifierTrainer(model = model, model_name = model_name,
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
    run_model_training(args = script_arguments)
