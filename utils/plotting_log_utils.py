import matplotlib.pyplot as plt
import numpy as np
import os

def plot_figure(train_stats, val_stats, legends, x_label, y_label, title, figure_path, show_image):
    assert len(train_stats) == len(val_stats)
    num_epochs = len(train_stats)
    epochs = np.array([i for i in range(1, num_epochs + 1)])

    fig = plt.figure(figsize=(5, 5))
    plt.plot(epochs, train_stats, 'b--')
    plt.plot(epochs, val_stats, 'r-')
    plt.legend(legends, fontsize = 'x-large')
    plt.xlabel(x_label, fontsize = 'x-large')
    plt.ylabel(y_label, fontsize = 'x-large')
    plt.title(title, fontsize = 'xx-large')

    if figure_path is not None:
        plt.savefig(figure_path)

    if show_image:
        plt.show()

    plt.close(fig = fig)

def plot_loss(model_name, train_loss_history, val_loss_history, directory, show_image = False):
    plot_legends = ["Training loss", "Validation loss"]
    plot_title = model_name + " loss per epoch"

    x_label = "Epochs"
    y_label = "Loss"

    if directory is not None:
        plot_file_name = model_name + "_loss_per_epoch.png"
        plot_path = os.path.join(directory, plot_file_name)

    else:
        plot_path = None

    plot_figure(train_stats = train_loss_history,
                val_stats = val_loss_history,
                legends = plot_legends,
                x_label = x_label,
                y_label = y_label,
                title = plot_title,
                figure_path = plot_path,
                show_image = show_image)

def plot_accuracy(model_name, train_accuracy_history, val_accuracy_history, directory, show_image = False):
    plot_legends = ["Training accuracy", "Validation accuracy"]
    plot_title = model_name + " accuracy per epoch"

    x_label = "Epochs"
    y_label = "Accuracy"

    if directory is not None:
        plot_file_name = model_name + "_accuracy_per_epoch.png"
        plot_path = os.path.join(directory, plot_file_name)

    else:
        plot_path = None

    plot_figure(train_stats = train_accuracy_history,
                val_stats = val_accuracy_history,
                legends = plot_legends,
                x_label = x_label,
                y_label = y_label,
                title = plot_title,
                figure_path = plot_path,
                show_image = show_image)
