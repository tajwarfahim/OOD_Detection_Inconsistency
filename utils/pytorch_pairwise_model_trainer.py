# Citation:
# 1. https://thenewstack.io/tutorial-train-a-deep-learning-model-in-pytorch-and-export-it-to-onnx/
# 2. https://discuss.pytorch.org/t/creating-custom-dataset-from-inbuilt-pytorch-datasets-along-with-data-transformations/58270/2
# 3. https://medium.com/@sergioalves94/deep-learning-in-pytorch-with-cifar-10-dataset-858b504a6b54
# 4. https://github.com/fangpin/siamese-pytorch/blob/master/train.py
# 5. https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html

import torch
import time
import numpy as np
import os
import shutil
import copy

def print_message(message, verbose = True):
    if verbose:
        print("")
        print(message)
        print("")

# should_use_scheduler -> True, use scheduler at every batch (instead of every epoch) like Outlier-exposure paper
# should_use_scheduler -> False, use scheduler at every epoch
def train_model(model, train_loader, optimizer, scheduler, should_use_scheduler):
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction = 'mean')
    model.train()

    for idx, (img1, img2, label) in enumerate(train_loader):
        if torch.cuda.is_available():
            img1 = img1.cuda()
            img2 = img2.cuda()
            label = label.cuda()

        model.zero_grad()
        output = model.forward(img1, img2).squeeze()
        loss = loss_fn(output, label.float())

        loss.backward()
        optimizer.step()

        if should_use_scheduler and scheduler is not None:
            scheduler.step()

def test_model(model, test_loader):
    model.eval()
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction = 'sum')

    total_loss = 0
    num_correct = 0
    num_total = 0

    with torch.no_grad():
        for idx, (img1, img2, label) in enumerate(test_loader):
            if torch.cuda.is_available():
                img1 = img1.cuda()
                img2 = img2.cuda()
                label = label.cuda()

            output = model.forward(img1, img2).squeeze()
            loss = loss_fn(output, label.float())
            total_loss += loss.item()

            pred = (output > 0).long()
            num_correct += (pred.squeeze() == label.squeeze()).float().sum().item()
            num_total += pred.shape[0]

    accuracy = float(num_correct) / num_total
    average_loss = float(total_loss) / num_total
    return average_loss, accuracy

class PairwiseModelTrainer:
    def __init__(self, model, model_name, train_loader, validation_loader, optimizer, scheduler = None):
        self.model = model
        self.model_name = model_name

        self.train_loader = train_loader
        self.validation_loader = validation_loader

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.max_val_accuracy = None
        self.best_epoch = None

        self.train_loss_history = []
        self.train_accuracy_history = []
        self.val_loss_history = []
        self.val_accuracy_history = []
        self.time_history = []

    def run_training(self, num_epochs, model_path = None, verbose = True, should_use_scheduler = False):
        print_message(message = "Model training is starting...", verbose = verbose)

        best_model_params = None
        for epoch in range(1, num_epochs + 1):
            start_time = time.time()

            # run the training step
            train_model(model = self.model, train_loader = self.train_loader, optimizer = self.optimizer, scheduler = self.scheduler, should_use_scheduler = should_use_scheduler)

            # calculate the training and validation loss and accuracy
            train_loss, train_accuracy = test_model(model = self.model, test_loader = self.train_loader)
            val_loss, val_accuracy = test_model(model = self.model, test_loader = self.validation_loader)

            end_time = time.time()

            if not should_use_scheduler and self.scheduler is not None:
                self.scheduler.step()

            self.train_loss_history.append(train_loss)
            self.train_accuracy_history.append(train_accuracy)
            self.val_loss_history.append(val_loss)
            self.val_accuracy_history.append(val_accuracy)

            time_per_epoch = end_time - start_time
            self.time_history.append(time_per_epoch)

            if verbose:
                print("Epoch: ", epoch, "/", num_epochs, " Train Accuracy: ", train_accuracy, " Val Accuracy: ", val_accuracy)
                print("            Train loss: ", train_loss, " Val loss: ", val_loss)
                print("Learning rate: ", self.scheduler.get_last_lr()[0])
                print()

            if epoch == 1 or val_accuracy > self.max_val_accuracy:
                self.best_epoch = epoch
                self.max_val_accuracy = val_accuracy
                best_model_params = copy.deepcopy(self.model.state_dict())

        if model_path is not None:
            # torch.save(best_model_params, model_path)
            torch.save(self.model.state_dict(), model_path)

        if verbose:
            total_time = 0
            for time_epoch in self.time_history:
                total_time += time_epoch

            print()
            print("Total training time: ", total_time, " seconds")
            print()

    def report_peak_performance(self):
        if self.max_val_accuracy == None:
            print("Model has not been trained yet.")
        else:
            print()
            print("Model peaked in validation accuracy at epoch ", self.best_epoch)
            print("Model peak validation accuracy: ", self.max_val_accuracy)
            print()

    def save_log(self, log_dir):
        print_message("Log directory: " + log_dir, verbose = 1)

        record_list = [self.train_loss_history, self.val_loss_history, self.train_accuracy_history, self.val_accuracy_history]
        filename_list = ["train_loss", "val_loss", "acc", "val_acc"]
        filename_prefix = "_per_epoch.txt"

        for i in range(len(record_list)):
            numpy_record = np.array(record_list[i])
            filename = filename_list[i] + filename_prefix
            filepath = os.path.join(log_dir, filename)
            np.savetxt(filepath, numpy_record, delimiter=",", fmt = "%1.4e")

        timer_log = np.array(self.time_history)
        total_time = np.sum(timer_log) / 3600.0

        appended =  [total_time] + self.time_history
        appended_log = np.array(appended)
        timer_path = os.path.join(log_dir, "training_time.txt")
        np.savetxt(timer_path, appended_log, delimiter=",", fmt = "%1.4e")
