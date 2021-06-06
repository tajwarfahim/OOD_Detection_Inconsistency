# Lack of Consistency Between OOD Detection Methods

This repository contains the official code for the ICML UDL Workshop 2021 Submission: "No True State-of-the-Art? OOD Detection Methods are Inconsistent across Datasets" by Fahim Tajwar, Ananya Kumar, Sang Michael Xie and Percy Liang.

Any correspondence should be addressed to Fahim Tajwar (tajwarfahim932@gmail.com or tajwar93@stanford.edu).

## Acknowledgements
We gratefully acknowledge authors of the following repositories (and give appropriate citation in our submission):

1. [Outlier Exposure](https://github.com/hendrycks/outlier-exposure),
2. [Energy based Out-of-distribution Detection](https://github.com/wetliu/energy_ood),
3. [Mahalanobis Method for OOD Detection](https://github.com/pokaxpoka/deep_Mahalanobis_detector) and
4. [Siamese Network](https://github.com/fangpin/siamese-pytorch)

as we integrate part of their code for our own work. Furthermore, neural network architectures are adopted from [Outlier exposure](https://github.com/hendrycks/outlier-exposure) and [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar). All other minor documentation help are cited in individual python scripts.

## Setup Conda environment for the experiments

Here we describe setting up the Conda environment and installing the dependencies to run our experiment. Use the following commands in shell.

```
# created on 2019-03-07

# Python version: 3.6.8
# PyTorch version: 1.0.1.post2
# TensorFlow version: 1.12.0

# create the environment
conda create -n py-3.6.8 python=3.6.8

# install tensorflow
conda install tensorflow-gpu=1.12.0

# downgrade cudnn to work on the cluster
conda install cudnn=7.1.2

# install pytorch
conda install pytorch torchvision -c pytorch
```

## Download datasets

In order to download the standard CIFAR-10, CIFAR-100, SVHN, CelebA and STL-10 datasets that we use in our work, use our "download_datasets.py" script. Use the following command in a shell.

```
python download_datasets.py --dataset_path_prefix "./data/" --dataset_name "all"
```

Or if the user wants to download only one of "CIFAR10", "CIFAR100", "SVHN", "CelebA" or "STL10", use the following command:

```
python download_datasets.py --dataset_path_prefix "./data/" --dataset_name "dataset_name"
```

Note that in our work we also use TinyImageNet (resized) and LSUN (resized) datasets. In order to download these, please follow the instruction in this repository: [ODIN: Out-of-Distribution Detector for Neural Networks](https://github.com/facebookresearch/odin).

Note that all the scripts below assume the individual datasets used for training/testing has been pre-downloaded, and may not try to download datasets themselves. Please ensure you have the proper environment set up and the datasets downloaded. Also, all of the following code assumes one GPU is available, and the code is not guaranteed to work without any GPU. Please make necessary changes in the code if this is the case.

## Train Models

Here is a sample shell command one can use to train a regular classifier:

```
python ./train_classifier.py --lr 0.1 --base_rate 1.0 --num_classes 10 --dataset_name CIFAR10 --dataset_path ./data/CIFAR10/ --model_type ResNet34 --num_epochs 200 --use_default_scheduler 0 --train_batch_size 128 --model_save_path ./Saved_Models/ --log_directory ./training_logs/classifier/
```

Here is a sample shell command to train a pairwise classifier from scratch:

```
python ./train_pairwise_classifier.py --base_rate 1.0 --dataset_name CIFAR10 --dataset_path ./data/CIFAR10/ --num_epochs 200 --lr 0.01 --shared_model_name ResNet34 --version 2 --train_epoch_size 25000 --use_default_scheduler 0 --train_batch_size 32 --model_save_path ./Saved_Models/ --projection_dim 512
```

Here is a sample shell command to train a pairwise classifier by fine-tuning a pre-trained classifier:

```
python ./train_pairwise_classifier.py --base_rate 1.0 --dataset_name CIFAR10 --num_classes 10 --dataset_path ./data/CIFAR10/ --num_epochs 25 --lr 0.01 --shared_model_name ResNet34 --version 2 --train_epoch_size 25000 --use_default_scheduler 0 --train_batch_size 32 --model_save_path ./Saved_Models/ --projection_dim 512 --presaved_model_path ./Saved_Models/model_weights/ --fine_tune 1
```

Finally, in order to change the number of ID train examples available to a model during training, use the "--base_rate" argument. This argument takes a float as an input between 0 and 1, and the script uses (base_rate * number of ID train examples) examples for training.

## Test Models

One can use the following command to use a regular classifier trained to evaluate the performance of MSP, ODIN and Mahalanobis (code used mainly from [Energy based OOD Detection](https://github.com/hendrycks/outlier-exposure) to implement these baselines) methods on a particular (ID, OOD) pair.

```
python ./test_classifier.py --num_classes 10 --id_dataset_name CIFAR10 --id_dataset_path ./data/CIFAR10/ --ood_dataset_name CIFAR100 --ood_dataset_path ./data/CIFAR100/ --model_type ResNet34 --presaved_model_path ./Saved_Models/model_weights/ --batch_size 200 --id_base_rate 1.0
```

where id_base_rate is the base_rate the model was used to train on.

In order to test a CP model, use the following command:

```
python ./test_pairwise_classifier.py --num_classes 10 --id_dataset_name CIFAR10 --id_dataset_path ./data/CIFAR10/ --siamese_network_version 5 --use_full_dataset -1 --should_plot 0 --ood_dataset_name CIFAR100 --ood_dataset_path ./data/CIFAR100/ --shared_model_name ResNet34 --presaved_model_path ./Saved_Models/model_weights/ --name_accumulator_across_classes "min" --name_accumulator_across_examples_in_a_class "average" --projection_dim 512 --use_first_samples 1 --default_sample_size 20 --fine_tune 0
```

In order to test a CP + Fine-tune model, just use the argument "--fine_tune 1" in the above script.

## Lack of consistency among MSP, ODIN and Mahalanobis baselines

The following table (taken from our paper) demonstrates the lack of consistency among these popular OOD detection baselines:

![image](https://github.com/tajwarfahim/OOD_Detection_Inconsistency/blob/main/figure/inconsistency_table.png)
