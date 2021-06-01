from __future__ import print_function
import torch
import numpy as np
import time
import argparse
import os
import matplotlib.pyplot as plt
import sys

# imports from our scripts
from utils.pairwise_ood_metrics import OODMetricsCalculator
from utils.ood_scoring_scheme import OODScoreCalculator

from utils.siamese_network import create_siamese_network_wrapper
from utils.wide_resnet_pytorch import create_wide_resnet
from utils.resnet_pytorch import ResNet18, ResNet34, ResNet50

from utils.pytorch_pairwise_dataset import PreSavedPerClassSampledDataset
from utils.pytorch_pairwise_dataset import RandomlySampledDataset
from utils.pytorch_pairwise_dataset import SelectiveClassDataset
from utils.pytorch_pairwise_dataset import load_dataset
from utils.pytorch_pairwise_dataset import check_dataset

def parse_arguments():
    ap = argparse.ArgumentParser()

    # -1 -> use one sample size, the default is set by another argument
    # 0 -> use all the partial datasets
    # 1 -> use the full dataset
    ap.add_argument("-use_full_dataset", "--use_full_dataset", type = int, default = -1, choices = [-1, 0, 1])
    ap.add_argument("-default_sample_size", "--default_sample_size", type = int, default = 20, choices = [1, 2, 5, 10, 20])
    ap.add_argument("-siamese_network_version", "--siamese_network_version", type = int, default = 2, choices = [1, 2, 3, 4, 5])
    ap.add_argument("-temp", "--temp", type = float, default = 0.1)
    ap.add_argument("-projection_dim", "--projection_dim", type = int, default = 512)

    # ID dataset
    ap.add_argument("-id_dataset_name", "--id_dataset_name", type = str, default = "CIFAR10")
    ap.add_argument("-id_dataset_path", "--id_dataset_path", type = str)
    ap.add_argument("-id_base_rate", "--id_base_rate", type = float, default = 1.0)
    ap.add_argument("-num_classes", "--num_classes", type = int, default = 10)
    ap.add_argument("-sample_indices_path_prefix", "--sample_indices_path_prefix", type = str, default = "./")

    # use first few samples per class
    ap.add_argument("-use_first_samples", "--use_first_samples", type = int, default = 0, choices = [0, 1])

    # OOD dataset
    ap.add_argument("-ood_dataset_name", "--ood_dataset_name", type = str, default = "CIFAR100")
    ap.add_argument("-ood_dataset_path", "--ood_dataset_path", type = str)

    # Model info
    ap.add_argument("-shared_model_name", "--shared_model_name", type = str, default = "ResNet34")
    ap.add_argument("-presaved_model_path", "--presaved_model_path", type = str)

    # metric info
    ap.add_argument("-recall_level", "--recall_level", type = float, default = 0.95)
    ap.add_argument("-tpr", "--tpr", type = float, default = 0.95)
    ap.add_argument("-fpr", "--fpr", type = float, default = 0.05)

    # Misc
    ap.add_argument("-num_workers", "--num_workers", type = int, default = 4)
    ap.add_argument("-batch_size", "--batch_size", type = int, default = 1024)
    ap.add_argument("-verbose", "--verbose", type = bool, default = True)
    ap.add_argument("-embedding_save_directory", "--embedding_save_directory", type = str, default = None)
    ap.add_argument("-threshold_for_accuracy", "--threshold_for_accuracy", type = float, default = None)
    ap.add_argument("-num_divides_of_ID_dataset", "--num_divides_of_ID_dataset", type = int, default = 50)
    ap.add_argument("-outlier_exposure", "--outlier_exposure", type = int, default = 1, choices = [0, 1])

    # plotting info
    ap.add_argument("-plot_save_path", "--plot_save_path", type = str)

    # if should_plot == 0: does not plot
    # if should plot == 1: only plot the auroc and aupr curves
    # if should_plot == 2: plot everything
    ap.add_argument("-should_plot", "--should_plot", type = int, default = 0, choices = [0, 1, 2])

    # accumulator function
    ap.add_argument("-name_accumulator_across_classes", "--name_accumulator_across_classes", type = str, default = "average", choices = ["max", "average", "min", "max - min", "all"])
    ap.add_argument("-name_accumulator_across_examples_in_a_class", "--name_accumulator_across_examples_in_a_class", type = str, default = "average", choices = ["max", "average", "min", "max - min", "all"])
    ap.add_argument("-accumulation_order", "--accumulation_order", type = str, default = "across_examples_first", choices = ["across_examples_first", "across_classes_first"])

    # use selected classes
    ap.add_argument("-ood_selective_class", "--ood_selective_class", type = int, default = 0, choices = [0, 1])
    ap.add_argument("-ood_labels_path", "--ood_labels_path", type = str, default = "./")

    ap.add_argument("-id_selective_class", "--id_selective_class", type = int, default = 0, choices = [0, 1])
    ap.add_argument("-id_labels_path", "--id_labels_path", type = str, default = "./")

    # fine tune on a pretrained classifier
    ap.add_argument("-fine_tune", "--fine_tune", type = int, default = 0, choices = [0, 1, 2])

    script_arguments = vars(ap.parse_args())
    return script_arguments

def print_message(message):
    print()
    print(message)
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
    architecture_map = process_shared_model_name_wide_resnet(shared_model_name = args["shared_model_name"])
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
    if args["shared_model_name"].find("WideResNet") == 0:
        model = create_wide_resnet_model(args)

    elif args["shared_model_name"].find("ResNet") >= 0:
        model = create_resnet_model(args)

    else:
        raise ValueError("Given model type is not supported")

    return model

def prepare_model(args):
    model = create_siamese_network_wrapper(dataset_name = args["id_dataset_name"],
                                           shared_model_name = args["shared_model_name"],
                                           load_to_gpu = False,
                                           version = args["siamese_network_version"],
                                           temp = args["temp"],
                                           projection_dim = args["projection_dim"],
                                           num_classes = args["num_classes"])

    if args["fine_tune"] > 0:
        classifier_model = create_classifier_model(args = args)
        model.shared_model = classifier_model

    print_message(message = "Model architecture built.")
    print(model)

    if args["siamese_network_version"] == [1, 2, 3, 4]:
        model.load_state_dict(torch.load(f = args["presaved_model_path"], map_location = torch.device("cpu")))

    elif args["siamese_network_version"] == 5:
        if args["fine_tune"] > 0:
            model.load_state_dict(torch.load(f = args["presaved_model_path"], map_location = torch.device("cpu")))
        else:
            model.shared_model.load_state_dict(torch.load(f = args["presaved_model_path"], map_location = torch.device("cpu")))

    else:
        raise ValueError("Given arguments are not compatible with the program.")

    print_message(message = "Pre-trained model parameters loaded.")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        model.to(device)
        print_message(message = "Model loaded to gpu.")

    return model

# we expect the inputs to have certain features, in accordance with the
# conventions we follow in this experiment
# we leverage those conventions here for simplicity
def prepare_experiment_plot_directory(args, name_accumulator_across_examples_in_a_class, name_accumulator_across_classes):
    model_timestamp_str = args["presaved_model_path"].split("/")[-1]
    dataset_names = "id_" + args["id_dataset_name"] + "_ood_" + args["ood_dataset_name"]
    function_names = "accumulator_across_examples=" + name_accumulator_across_examples_in_a_class + "_accumulator_across_classes=" + name_accumulator_across_classes

    plot_directory = os.path.join(args["plot_save_path"],
                                  model_timestamp_str,
                                  dataset_names,
                                  function_names)

    if not os.path.isdir(plot_directory):
        os.makedirs(plot_directory)

    return plot_directory

def prepare_metrics_for_plotting(metric_name, performance_record, name_accumulator_across_examples_in_a_class, name_accumulator_across_classes):
    lst = []
    sample_sizes = [sample_size for sample_size in performance_record]
    sample_sizes.sort()

    for sample_size in sample_sizes:
        value_of_metric = performance_record[sample_size][name_accumulator_across_examples_in_a_class][name_accumulator_across_classes][metric_name]
        lst.append(value_of_metric)

    lst = np.array(lst, dtype = np.float64)
    sample_sizes = np.sort(np.array(sample_sizes, dtype = np.int32))

    return lst, sample_sizes

def plot_figure(x, y, x_label, y_label, title, figure_path):
    assert len(x.shape) == len(y.shape)
    assert len(x.shape) == 1
    assert x.shape[0] == y.shape[0]

    fig = plt.figure(figsize=(5, 5))
    plt.plot(x, y, 'r-')
    plt.xlabel(x_label, fontsize = 'x-large')
    plt.ylabel(y_label, fontsize = 'x-large')
    plt.title(title, fontsize = 'xx-large')

    if figure_path is not None:
        plt.savefig(figure_path)

    plt.close(fig = fig)

def plot_experiment_results(args, performance_record, metric_names, names_accumulator_across_examples_in_a_class, names_accumulator_across_classes):
    print("Metric names: ", metric_names)

    for name_accumulator_across_examples_in_a_class in names_accumulator_across_examples_in_a_class:
        for name_accumulator_across_classes in names_accumulator_across_classes:
            plot_directory = prepare_experiment_plot_directory(args = args,
                                                               name_accumulator_across_examples_in_a_class = name_accumulator_across_examples_in_a_class,
                                                               name_accumulator_across_classes = name_accumulator_across_classes)

            print_message(message = "Plots saved in : " + plot_directory)

            for metric_name in metric_names:
                file_name = metric_name + ".png"
                figure_path = os.path.join(plot_directory, file_name)

                metrics, sample_sizes = prepare_metrics_for_plotting(metric_name = metric_name,
                                                                     performance_record = performance_record,
                                                                     name_accumulator_across_examples_in_a_class = name_accumulator_across_examples_in_a_class,
                                                                     name_accumulator_across_classes = name_accumulator_across_classes)

                plot_figure(x = sample_sizes,
                            y = metrics,
                            x_label = "Sample size",
                            y_label = metric_name,
                            title = metric_name + " vs sample size",
                            figure_path = figure_path)

def print_arguments(args):
    if args["verbose"]:
        print()
        print("Arguments given for the script...")
        for key in args:
            print("Key: ", key, " Value: ", args[key])
        print()

def print_sample_sizes(args, sample_sizes):
    if args["verbose"]:
        print()
        print("Sample sizes we experiment on: ", sample_sizes)
        print()

def get_id_and_ood_test_dataloaders(args):
    id_test_dataset = load_dataset(dataset_name = args["id_dataset_name"],
                                   dataset_path = args["id_dataset_path"],
                                   train = False,
                                   id_dataset_name = args["id_dataset_name"],
                                   id_dataset_path = args["id_dataset_path"],
                                   augment = False)

    ood_test_dataset = load_dataset(dataset_name = args["ood_dataset_name"],
                                    dataset_path = args["ood_dataset_path"],
                                    train = False,
                                    id_dataset_name = args["id_dataset_name"],
                                    id_dataset_path = args["id_dataset_path"],
                                    augment = False)

    if args["id_selective_class"] == 1:
        id_test_dataset = SelectiveClassDataset(dataset = id_test_dataset, labels_path = args["id_labels_path"])

    if args["ood_selective_class"] == 1:
        ood_test_dataset = SelectiveClassDataset(dataset = ood_test_dataset, labels_path = args["ood_labels_path"])

    # # fix the base rate
    # id_test_dataset = RandomlySampledDataset(dataset = id_test_dataset, base_rate = args["id_base_rate"])
    # ood_test_dataset = RandomlySampledDataset(dataset = ood_test_dataset, base_rate = args["ood_base_rate"])

    id_test_dataloader = torch.utils.data.DataLoader(dataset = id_test_dataset, num_workers = args["num_workers"], batch_size = args["batch_size"], shuffle = False)
    ood_test_dataloader = torch.utils.data.DataLoader(dataset = ood_test_dataset, num_workers = args["num_workers"], batch_size = args["batch_size"], shuffle = False)

    return id_test_dataloader, ood_test_dataloader

def get_id_train_dataloaders(args, sample_size_lst):
    id_train_dataset = load_dataset(dataset_name = args["id_dataset_name"],
                                    dataset_path = args["id_dataset_path"],
                                    train = True,
                                    id_dataset_name = args["id_dataset_name"],
                                    id_dataset_path = args["id_dataset_path"],
                                    augment = False)

    sample_indices_filenames = {}
    for sample_size in sample_size_lst:
        filename = args["sample_indices_path_prefix"] + "=" + str(sample_size) + ".txt"
        # assert os.path.isfile(filename)
        sample_indices_filenames[sample_size] = filename

    id_train_dataloaders = {}
    for sample_size in sample_size_lst:
        # choose a certain per
        if args["use_first_samples"] > 0:
            dataset = RandomlySampledDataset(dataset = id_train_dataset,
                                             base_rate = 1.0,
                                             choose_randomly = False,
                                             fixed_sample_size = sample_size)

        elif args["id_base_rate"] < 1.0:
            dataset = RandomlySampledDataset(dataset = id_train_dataset,
                                             base_rate = args["id_base_rate"],
                                             choose_randomly = False)

        elif args["id_selective_class"] == 0:
            dataset = PreSavedPerClassSampledDataset(dataset = id_train_dataset,
                                                     sample_indices_path = sample_indices_filenames[sample_size])

        else:
            selective_class_dataset = SelectiveClassDataset(dataset = id_train_dataset,
                                                            labels_path = args["id_labels_path"])

            dataset = PreSavedPerClassSampledDataset(dataset = selective_class_dataset,
                                                     sample_indices_path = sample_indices_filenames[sample_size])

        dataloader = torch.utils.data.DataLoader(dataset = dataset, num_workers = args["num_workers"], batch_size = args["batch_size"], shuffle = False)
        id_train_dataloaders[sample_size] = dataloader

    return id_train_dataloaders

def prepare_embeddings_for_the_experiment(ood_score_calculator, args, device_type):
    print_message(message = "Starting to calculate embeddings...")

    start_time = time.time()
    ood_score_calculator.prepare_embeddings(save_directory = args["embedding_save_directory"], device_type = device_type)
    end_time = time.time()

    total_time = end_time - start_time
    print("It took ", total_time, " seconds to pre-calculate embeddings")

    print_message(message = "Embeddings have been calculated.")

def prepare_pairwise_comparison_scores_for_the_experiment(ood_score_calculator, args):
    print_message(message = "Starting pairwise comparison between ID train and test sets (ID test and OOD test)...")

    start_time = time.time()

    ood_score_calculator.run_pairwise_comparison(siamese_network_version = args["siamese_network_version"],
                                                 num_divides_of_ID_dataset = args["num_divides_of_ID_dataset"],
                                                 temp = args["temp"])

    end_time = time.time()

    total_time = end_time - start_time
    print("It took ", total_time, " seconds to run pairwise comparison")

    print_message(message = "Pairwise comparison finished.")

# this currently only prints metrics that we want, add additional if you want
def print_metrics(args, perfomance_metrics):
    print()
    print("Accumulator across ID classes: ", perfomance_metrics["name_accumulator_across_classes"])
    print("Accumulator across examples in a class: ", perfomance_metrics["name_accumulator_across_examples_in_a_class"])

    print("AUROC: ", perfomance_metrics["AUROC"])
    print("AUPR-In: ", perfomance_metrics["AUPR-In"])
    print("AUPR-Out: ", perfomance_metrics["AUPR-Out"])
    print("ID test classification accuracy: ", perfomance_metrics["classification_accuracy"])
    print("FPR at recall recall level ", args["recall_level"], ": ", perfomance_metrics["fpr"])

    print("ID score average: ", perfomance_metrics["id_average"])
    print("OOD score average: ", perfomance_metrics["ood_average"])
    print()

def plot_auroc_and_aupr(args, metrics_calculator, name_accumulator_across_examples_in_a_class, name_accumulator_across_classes, sample_size):
    plot_directory = prepare_experiment_plot_directory(args = args,
                                                       name_accumulator_across_examples_in_a_class = name_accumulator_across_examples_in_a_class,
                                                       name_accumulator_across_classes = name_accumulator_across_classes)

    plot_subdirectory = os.path.join(plot_directory, "sample_size_per_ID_class=" + str(sample_size))
    if not os.path.isdir(plot_subdirectory):
        os.makedirs(plot_subdirectory)

    auroc_file_name = "AUROC.png"
    aupr_file_name = "AUPR.png"

    auroc_file_path = os.path.join(plot_subdirectory, auroc_file_name)
    aupr_file_path = os.path.join(plot_subdirectory, aupr_file_name)

    metrics_calculator.draw_model_roc_curve(name_accumulator_across_classes = name_accumulator_across_classes, save_path = auroc_file_path)
    metrics_calculator.draw_model_pr_curve(name_accumulator_across_classes = name_accumulator_across_classes, save_path = aupr_file_path)

    print()
    print("Model ROC and PR curves plotted in: ", plot_subdirectory)
    print()


def run_a_single_experiment(args, ood_score_calculator, name_accumulator_across_examples_in_a_class, name_accumulator_across_classes, sample_size):
    classification_accuracy = ood_score_calculator.calculate_id_test_classification_accuracy(name_accumulator_across_examples_in_a_class = name_accumulator_across_examples_in_a_class,
                                                                                             sample_size_per_class = sample_size)

    id_scores, ood_scores = ood_score_calculator.get_id_and_ood_score(name_accumulator_across_examples_in_a_class = name_accumulator_across_examples_in_a_class,
                                                                      name_accumulator_across_classes =  name_accumulator_across_classes,
                                                                      sample_size_per_class = sample_size,
                                                                      order = args["accumulation_order"],
                                                                      return_scores_after_class_reduction = False)

    metrics_calculator = OODMetricsCalculator(id_scores = id_scores,
                                              ood_scores = ood_scores,
                                              id_dataset_name = args["id_dataset_name"],
                                              ood_dataset_name = args["ood_dataset_name"]) #,
                                              #outlier_exposure = args["outlier_exposure"])

    id_average = np.average(id_scores)
    ood_average = np.average(ood_scores)

    auroc = metrics_calculator.get_roc_auc_score(name_accumulator_across_classes = name_accumulator_across_classes)
    aupr_in = metrics_calculator.get_pr_auc_score(name_accumulator_across_classes = name_accumulator_across_classes, positive = "in")
    aupr_out = metrics_calculator.get_pr_auc_score(name_accumulator_across_classes = name_accumulator_across_classes, positive = "out")
    fpr = metrics_calculator.get_fpr(name_accumulator_across_classes = name_accumulator_across_classes, recall_level = args["recall_level"])

    perfomance_metrics = {"AUROC": auroc, "AUPR-In": aupr_in, "AUPR-Out": aupr_out,
                          "name_accumulator_across_examples_in_a_class": name_accumulator_across_examples_in_a_class,
                          "name_accumulator_across_classes": name_accumulator_across_classes,
                          "id_average": id_average, "ood_average": ood_average,
                          "classification_accuracy": classification_accuracy, "fpr" : fpr}

    print_metrics(args = args, perfomance_metrics = perfomance_metrics)

    if args["should_plot"] > 0:
        plot_auroc_and_aupr(args = args,
                            metrics_calculator = metrics_calculator,
                            name_accumulator_across_examples_in_a_class = name_accumulator_across_examples_in_a_class,
                            name_accumulator_across_classes = name_accumulator_across_classes,
                            sample_size = sample_size)

    return perfomance_metrics

def run_experiment_over_different_comparison_functions(args, sample_size, model, id_train_dataloader, id_test_dataloader, ood_test_dataloader, device_type,
                                                       names_accumulator_across_examples_in_a_class, names_accumulator_across_classes):

    ood_score_calculator = OODScoreCalculator(siamese_network = model,
                                              id_train_dataloader = id_train_dataloader,
                                              id_test_dataloader = id_test_dataloader,
                                              ood_dataloader = ood_test_dataloader)

    prepare_embeddings_for_the_experiment(ood_score_calculator = ood_score_calculator, args = args, device_type = device_type)
    prepare_pairwise_comparison_scores_for_the_experiment(ood_score_calculator = ood_score_calculator, args = args)

    performance_record_per_accumulator_across_examples_in_a_class = {}
    for name_accumulator_across_examples_in_a_class in names_accumulator_across_examples_in_a_class:
        performance_record_per_accumulator_across_classes = {}
        for name_accumulator_across_classes in names_accumulator_across_classes:
            performance_record_per_accumulator_across_classes[name_accumulator_across_classes] = run_a_single_experiment(args = args,
                                                                                                                         ood_score_calculator = ood_score_calculator,
                                                                                                                         name_accumulator_across_examples_in_a_class = name_accumulator_across_examples_in_a_class,
                                                                                                                         name_accumulator_across_classes = name_accumulator_across_classes,
                                                                                                                         sample_size = sample_size)

        performance_record_per_accumulator_across_examples_in_a_class[name_accumulator_across_examples_in_a_class] = performance_record_per_accumulator_across_classes

    return performance_record_per_accumulator_across_examples_in_a_class


def run_experiment_over_different_sample_sizes(args, model, id_train_dataloaders, id_test_dataloader, ood_test_dataloader, device_type, sample_size_lst,
                                               names_accumulator_across_examples_in_a_class, names_accumulator_across_classes):

    performance_record_per_sample_size = {}
    for sample_size in sample_size_lst:
        print()
        print("Sample size per ID class: ", sample_size)
        print()

        performance_record_per_sample_size[sample_size] = run_experiment_over_different_comparison_functions(args = args,
                                                                                                             sample_size = sample_size,
                                                                                                             model = model,
                                                                                                             id_train_dataloader = id_train_dataloaders[sample_size],
                                                                                                             id_test_dataloader = id_test_dataloader,
                                                                                                             ood_test_dataloader = ood_test_dataloader,
                                                                                                             device_type = device_type,
                                                                                                             names_accumulator_across_examples_in_a_class = names_accumulator_across_examples_in_a_class,
                                                                                                             names_accumulator_across_classes = names_accumulator_across_classes)

    return performance_record_per_sample_size


def run_experiments(args, sample_size_lst, names_accumulator_across_examples_in_a_class, names_accumulator_across_classes):
    # loading stuff
    id_train_dataloaders = get_id_train_dataloaders(args = args, sample_size_lst = sample_size_lst)
    id_test_dataloader, ood_test_dataloader = get_id_and_ood_test_dataloaders(args = args)
    model = prepare_model(args = args)

    tpr_name = "TPR_" + str(int(args["fpr"] * 100))
    fpr_name = "FPR_" + str(int(args["tpr"] * 100))

    if torch.cuda.is_available():
        device_type = "cuda"
    else:
        device_type = "cpu"

    performance_record = run_experiment_over_different_sample_sizes(args = args,
                                                                    model = model,
                                                                    id_train_dataloaders = id_train_dataloaders,
                                                                    id_test_dataloader = id_test_dataloader,
                                                                    ood_test_dataloader = ood_test_dataloader,
                                                                    device_type = device_type,
                                                                    sample_size_lst = sample_size_lst,
                                                                    names_accumulator_across_examples_in_a_class = names_accumulator_across_examples_in_a_class,
                                                                    names_accumulator_across_classes = names_accumulator_across_classes)

    if args["threshold_for_accuracy"] is not None:
        metric_names = [tpr_name, fpr_name, "AUROC", "AUPR", "Accuracy"]
    else:
        metric_names = [tpr_name, fpr_name, "AUROC", "AUPR", "Accuracy_with_tpr_n_threshold", "Accuracy_with_fpr_n_threshold"]

    return performance_record, metric_names

def get_sample_size_lst(args):
    if args["id_selective_class"] == 1:
        assert args["use_full_dataset"] != 0

    if args["use_full_dataset"] == -1:
        default_sample_size = args["default_sample_size"]
        return [default_sample_size]
    elif args["use_full_dataset"] == 0:
        return [1, 2, 5, 10, 20]
    elif args["use_full_dataset"] == 1 and args["id_dataset_name"] == "CIFAR10":
        return [5000]
    elif args["use_full_dataset"] == 1 and args["id_dataset_name"] == "CIFAR100":
        return [500]
    else:
        raise ValueError("Arguments given are not supported")

def run_script(args):
    start_time = time.time()

    sample_size_lst = get_sample_size_lst(args = args)

    # function names
    if args["name_accumulator_across_examples_in_a_class"] == "all":
        names_accumulator_across_examples_in_a_class = ["average", "min", "max", "max - min"]
    else:
        names_accumulator_across_examples_in_a_class = [args["name_accumulator_across_examples_in_a_class"]]

    if args["name_accumulator_across_classes"] == "all":
        names_accumulator_across_classes = ["average", "min", "max", "max - min"]
    else:
        names_accumulator_across_classes = [args["name_accumulator_across_classes"]]

    print_arguments(args = args)
    print_sample_sizes(args = args, sample_sizes = sample_size_lst)

    performance_record, metric_names = run_experiments(args = args,
                                                       sample_size_lst = sample_size_lst,
                                                       names_accumulator_across_examples_in_a_class = names_accumulator_across_examples_in_a_class,
                                                       names_accumulator_across_classes = names_accumulator_across_classes)


    if script_arguments["should_plot"] == 2:
        assert script_arguments["use_full_dataset"] == 0
        print("Plotting peformance metrics vs sample size results...")

        plot_experiment_results(args = args,
                                performance_record = performance_record,
                                metric_names = metric_names,
                                names_accumulator_across_examples_in_a_class = names_accumulator_across_examples_in_a_class,
                                names_accumulator_across_classes = names_accumulator_across_classes)

        print()

    end_time = time.time()

    total_time = end_time - start_time
    print()
    print("Total time to run this experiment: ", total_time, " s")
    print()

if __name__ == '__main__':
    script_arguments = parse_arguments()
    run_script(args = script_arguments)
