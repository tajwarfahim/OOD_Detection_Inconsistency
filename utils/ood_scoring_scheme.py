# Help taken from:
# 1. https://stackoverflow.com/questions/61892957/using-additional-kwargs-with-a-custom-function-for-scipys-cdist-or-pdist
# 2. https://jbencook.com/pairwise-distance-in-numpy/
# 3. https://numpy.org/doc/stable/reference/generated/numpy.einsum.html
# 4. https://stackoverflow.com/questions/26089893/understanding-numpys-einsum
# 5. https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.softmax.html

# Optimized/vectorized version of the functions added

# imports
import torch
import os
import numpy as np
import scipy
import sklearn

# entropy is no longer supported
def entropy(scores, axis):
    assert isinstance(scores, np.ndarray)
    assert len(scores.shape) == 2

    softmax_scores = scipy.special.softmax(x = scores, axis = axis)
    log_softmax_scores = np.log(softmax_scores)

    assert scores.shape == softmax_scores.shape
    assert softmax_scores.shape == log_softmax_scores.shape

    # elementwise multilplication
    plogp_matrix = softmax_scores * log_softmax_scores

    assert scores.shape == plogp_matrix.shape

    sum_matrix = np.sum(a = plogp_matrix, axis = axis)
    entropy_matrix = sum_matrix * (-1.0)

    return entropy_matrix


# returns embeddings in a tensor format
def precalculate_embeddings(siamese_network, dataloader, return_tensor = True):
    siamese_network.eval()

    all_embeddings = []
    with torch.no_grad():
        for img, _ in dataloader:
            if torch.cuda.is_available():
                img = img.cuda()

            embeddings = siamese_network.get_embedding(img).detach().cpu().numpy()
            all_embeddings.append(embeddings)

    all_embeddings = np.concatenate(all_embeddings)
    if return_tensor:
        all_embeddings = torch.from_numpy(all_embeddings)

    # assertions to check if everything is behaving as expected
    assert len(all_embeddings.shape) == 2
    assert all_embeddings.shape[0] == len(dataloader.dataset)

    print("Embedding shape: ", all_embeddings.shape)

    return all_embeddings

def append_constant_value_column_to_matrix(matrix, value):
    assert len(matrix.shape) == 2

    column = None
    if value == 1:
        column = np.ones(shape = (matrix.shape[0], 1), dtype = matrix.dtype)
    elif value == 0:
        column = np.zeros(shape = (matrix.shape[0], 1), dtype = matrix.dtype)
    else:
        raise ValueError("Given value in column not supported")

    new_matrix = np.append(matrix, column, axis = 1)
    assert len(new_matrix.shape) == 2
    assert new_matrix.shape[0] == matrix.shape[0]
    assert new_matrix.shape[1] == (matrix.shape[1] + 1)

    return new_matrix

def calculate_pairwise_elementwise_absolute_distance_matrix(matrix_A, matrix_B, siamese_network_version):
    # assertions to check if inputs are valid
    assert isinstance(matrix_A, np.ndarray) and isinstance(matrix_B, np.ndarray)
    assert len(matrix_A.shape) == 2
    assert len(matrix_B.shape) == 2
    assert matrix_A.shape[1] == matrix_B.shape[1]

    if siamese_network_version == 4:
        distance_matrix = sklearn.metrics.pairwise.cosine_similarity(matrix_A, matrix_B)
        return distance_matrix

    # using numpy broadcasting to take the pairwise, elementwise, absolute distance.
    # returns a 3D numpy array
    distance_matrix = matrix_A[:, np.newaxis, :] - matrix_B[np.newaxis, :, :]

    if siamese_network_version == 1:
        return np.abs(distance_matrix)

    elif siamese_network_version in [2, 5]:
        return distance_matrix * distance_matrix

    elif siamese_network_version == 3:
        return distance_matrix

    else:
        raise ValueError("Siamese network version:", siamese_network_version, " is not supported, it has to be either 1, 2 or 3")

def recover_siamese_network_weights(siamese_network, siamese_network_version):
    siamese_net_weight, siamese_net_bias = siamese_network.extract_last_layer_weights()
    if siamese_network_version in [1, 2]:
        combined_weight = np.append(np.reshape(a = siamese_net_weight, newshape = (-1, )), np.reshape(a = siamese_net_bias, newshape = (-1,)), axis = 0)
        assert len(combined_weight.shape) == 1
    else:
        combined_weight = np.transpose(siamese_net_weight)

    return combined_weight

def pairwise_comparison_scores(siamese_network, embeddings_of_test_dataset, embeddings_of_ID_dataset, num_divides_of_ID_dataset, siamese_network_version, temp):
    # assertions to check if inputs are valid
    assert isinstance(embeddings_of_test_dataset, np.ndarray) and isinstance(embeddings_of_ID_dataset, np.ndarray)
    assert len(embeddings_of_test_dataset.shape) == 2
    assert len(embeddings_of_ID_dataset.shape) == 2
    assert embeddings_of_test_dataset.shape[1] == embeddings_of_ID_dataset.shape[1]
    assert int(embeddings_of_ID_dataset.shape[0]) % num_divides_of_ID_dataset == 0
    assert siamese_network_version in [1, 2, 3, 4, 5]

    # preparing the embeddings
    if siamese_network_version in [1, 2]:
        embeddings_of_test_dataset = append_constant_value_column_to_matrix(matrix = embeddings_of_test_dataset, value = 1)
        embeddings_of_ID_dataset = append_constant_value_column_to_matrix(matrix = embeddings_of_ID_dataset, value = 0)

    # preparing the weights
    if siamese_network_version in [1, 2, 3]:
        siamese_net_weight = recover_siamese_network_weights(siamese_network = siamese_network, siamese_network_version = siamese_network_version)

    scores = []
    each_division_len = int(embeddings_of_ID_dataset.shape[0] / num_divides_of_ID_dataset)

    for i in range(num_divides_of_ID_dataset):
        partial_ID_dataset_embedding = embeddings_of_ID_dataset[i * each_division_len : (i + 1) * each_division_len, :]
        partial_distance_matrix = calculate_pairwise_elementwise_absolute_distance_matrix(matrix_A = embeddings_of_test_dataset,
                                                                                          matrix_B = partial_ID_dataset_embedding,
                                                                                          siamese_network_version = siamese_network_version)

        if siamese_network_version in [1, 2]:
            partial_scores = np.einsum("ijk,k->ij", partial_distance_matrix, siamese_net_weight)
            scores.append(np.transpose(partial_scores))

        elif siamese_network_version == 3:
            partial_M_distance_matrix = np.einsum("ijk,kl->ijl", partial_distance_matrix, siamese_net_weight)
            partial_scores = np.einsum("ijk,ijk->ij", partial_distance_matrix, partial_M_distance_matrix)
            scores.append(np.transpose(partial_scores))

        elif siamese_network_version == 4:
            partial_scores = (-1.0 / temp) * partial_distance_matrix
            partial_scores = np.transpose(partial_scores)
            scores.append(partial_scores)

        elif siamese_network_version == 5:
            partial_scores = np.einsum("ijk->ij", partial_distance_matrix)
            scores.append(np.transpose(partial_scores))

        print(i + 1, "/", num_divides_of_ID_dataset, " partial scores calculated")


    scores = np.transpose(np.concatenate(scores))
    # scores = torch.sigmoid(torch.from_numpy(scores)).numpy()

    assert isinstance(scores, np.ndarray)
    assert len(scores.shape) == 2
    assert scores.shape[0] == embeddings_of_test_dataset.shape[0]
    assert scores.shape[1] == embeddings_of_ID_dataset.shape[0]

    print()
    print("Scores shape: ", scores.shape)
    print()

    return scores

def assign_accumulator_function_from_name(function_name):
    if function_name == "min":
        return np.min

    elif function_name == "max":
        return np.max

    elif function_name == "average":
        return np.average

    elif function_name == "entropy":
        return entropy

    else:
        raise ValueError("Fuction name not supported")

def first_reduction(scores_3D, name_accumulator, axis):
    assert axis in [1, 2]
    assert isinstance(scores_3D, np.ndarray)
    assert len(scores_3D.shape) == 3

    if name_accumulator == "max - min":
        max_scores = np.max(scores_3D, axis = axis)
        min_scores = np.min(scores_3D, axis = axis)
        scores_after_first_reduction = max_scores - min_scores

    else:
        first_accumulator = assign_accumulator_function_from_name(function_name = name_accumulator)
        scores_after_first_reduction = first_accumulator(scores_3D, axis = axis)

    return scores_after_first_reduction

def second_reduction(scores_2D, name_accumulator):
    assert isinstance(scores_2D, np.ndarray)
    assert len(scores_2D.shape) == 2

    if name_accumulator == "max - min":
        max_scores = np.max(scores_2D, axis = 1)
        min_scores = np.min(scores_2D, axis = 1)
        scores_after_second_reduction = max_scores - min_scores

    else:
        second_accumulator = assign_accumulator_function_from_name(function_name = name_accumulator)
        scores_after_second_reduction = second_accumulator(scores_2D, axis = 1)

    return scores_after_second_reduction

def process_comparison_scores(scores, name_accumulator_across_examples_in_a_class, name_accumulator_across_classes, sample_size_per_class, order = "across_examples_first"):
    # assertions to make sure "scores" have correct format
    assert isinstance(scores, np.ndarray)
    assert len(scores.shape) == 2
    assert scores.shape[1] % sample_size_per_class == 0

    num_test_examples = int(scores.shape[0])
    num_ID_classes = int(scores.shape[1] / sample_size_per_class)
    num_samples_per_ID_class = int(sample_size_per_class)

    # reshaping score to be 3-dimensional
    scores_3D = np.reshape(a = scores, newshape = (num_test_examples, num_ID_classes, num_samples_per_ID_class))

    if order == "across_examples_first":
        scores_after_first_reduction = first_reduction(scores_3D = scores_3D, name_accumulator = name_accumulator_across_examples_in_a_class, axis = 2)
        scores_after_second_reduction = second_reduction(scores_2D = scores_after_first_reduction, name_accumulator = name_accumulator_across_classes)

    elif order == "across_classes_first":
        scores_after_first_reduction = first_reduction(scores_3D = scores_3D, name_accumulator = name_accumulator_across_classes, axis = 1)
        scores_after_second_reduction = second_reduction(scores_2D = scores_after_first_reduction, name_accumulator = name_accumulator_across_examples_in_a_class)

    else:
        raise ValueError("Given order for accumulating scores is not supported")

    return scores_after_second_reduction, scores_after_first_reduction


def get_classification_accuracy(id_test_scores, id_test_dataloader, name_accumulator_across_examples_in_a_class, sample_size_per_class):
    id_test_dataset = id_test_dataloader.dataset

    num_test_examples = int(id_test_scores.shape[0])
    num_ID_classes = int(id_test_scores.shape[1] / sample_size_per_class)
    num_samples_per_ID_class = int(sample_size_per_class)

    # reshaping score to be 3-dimensional
    scores_3D = np.reshape(a = id_test_scores, newshape = (num_test_examples, num_ID_classes, num_samples_per_ID_class))
    accumulated_scores = first_reduction(scores_3D = scores_3D, name_accumulator = name_accumulator_across_examples_in_a_class, axis = 2)
    assert accumulated_scores.shape == (num_test_examples, num_ID_classes)

    # getting prediction
    preds = np.argmin(accumulated_scores, axis = 1)
    assert len(preds.shape) == 1
    assert preds.shape[0] == len(id_test_dataset)

    # getting number of correct predictions
    num_correct = 0
    for i in range(len(id_test_dataset)):
        _, label = id_test_dataset[i]
        if torch.is_tensor(label):
            label = label.item()

        if label == preds[i]:
            num_correct = num_correct + 1

    classification_accuracy = float(num_correct) / num_test_examples
    classification_accuracy = round(classification_accuracy * 100, 2)

    return classification_accuracy



def get_all_embeddings(siamese_network, id_train_dataloader, id_test_dataloader, ood_dataloader, save_directory, device_type):
    dataloader_map = {"id_train": id_train_dataloader,
                      "id_test": id_test_dataloader,
                      "ood": ood_dataloader}

    embeddings_map = {}
    for key in dataloader_map:
        embeddings_map[key] = precalculate_embeddings(siamese_network, dataloader_map[key], False)

    if save_directory is not None:
        if not os.path.isdir(save_directory):
            os.makedir(save_directory)

        for key in embeddings_map:
            save_path = os.path.join(save_directory, key + "_embedding.txt")
            np.savetxt(save_path, embeddings_map[key], delimiter=",")

    return embeddings_map

# summary class, i.e. encapsulates all the functionalities in this script
# fascilitates calculating using different comparison functions
class OODScoreCalculator:
    def __init__(self, siamese_network, id_train_dataloader, id_test_dataloader, ood_dataloader):
        self.siamese_network = siamese_network
        self.id_train_dataloader = id_train_dataloader
        self.id_test_dataloader = id_test_dataloader
        self.ood_dataloader = ood_dataloader

        # things that need to be calculated
        self.embeddings_map = None
        self.id_pairwise_comparison_scores = None
        self.ood_pairwise_comparison_scores = None

    def prepare_embeddings(self, save_directory, device_type):
        self.embeddings_map = get_all_embeddings(siamese_network = self.siamese_network,
                                                 id_train_dataloader = self.id_train_dataloader,
                                                 id_test_dataloader = self.id_test_dataloader,
                                                 ood_dataloader = self.ood_dataloader,
                                                 save_directory = save_directory,
                                                 device_type = device_type)

    def run_pairwise_comparison(self, num_divides_of_ID_dataset = 50, siamese_network_version = 2, temp = 0.1):
        assert self.embeddings_map is not None

        self.id_pairwise_comparison_scores = pairwise_comparison_scores(siamese_network = self.siamese_network,
                                                                        embeddings_of_test_dataset = self.embeddings_map["id_test"],
                                                                        embeddings_of_ID_dataset = self.embeddings_map["id_train"],
                                                                        num_divides_of_ID_dataset = num_divides_of_ID_dataset,
                                                                        siamese_network_version = siamese_network_version,
                                                                        temp = temp)

        self.ood_pairwise_comparison_scores = pairwise_comparison_scores(siamese_network = self.siamese_network,
                                                                         embeddings_of_test_dataset = self.embeddings_map["ood"],
                                                                         embeddings_of_ID_dataset = self.embeddings_map["id_train"],
                                                                         num_divides_of_ID_dataset = num_divides_of_ID_dataset,
                                                                         siamese_network_version = siamese_network_version,
                                                                         temp = temp)


    def calculate_id_test_classification_accuracy(self, name_accumulator_across_examples_in_a_class, sample_size_per_class):
        assert self.id_pairwise_comparison_scores is not None

        classification_accuracy = get_classification_accuracy(id_test_scores = self.id_pairwise_comparison_scores,
                                                              id_test_dataloader = self.id_test_dataloader,
                                                              name_accumulator_across_examples_in_a_class = name_accumulator_across_examples_in_a_class,
                                                              sample_size_per_class = sample_size_per_class)

        return classification_accuracy


    def get_id_and_ood_score(self, name_accumulator_across_examples_in_a_class, name_accumulator_across_classes, sample_size_per_class, order, return_scores_after_class_reduction = False):
        assert self.id_pairwise_comparison_scores is not None
        assert self.ood_pairwise_comparison_scores is not None

        assert name_accumulator_across_examples_in_a_class in ["average", "min", "max", "max - min"]
        assert name_accumulator_across_classes in ["average", "min", "max", "max - min"]

        # entropy is no longer supported
        # assert name_accumulator_across_classes in ["average", "min", "max", "entropy"]

        id_scores, id_scores_after_class_reduction = process_comparison_scores(scores = self.id_pairwise_comparison_scores,
                                                                               name_accumulator_across_examples_in_a_class = name_accumulator_across_examples_in_a_class,
                                                                               name_accumulator_across_classes = name_accumulator_across_classes,
                                                                               sample_size_per_class = sample_size_per_class,
                                                                               order = order)

        ood_scores, ood_scores_after_class_reduction = process_comparison_scores(scores = self.ood_pairwise_comparison_scores,
                                                                                 name_accumulator_across_examples_in_a_class = name_accumulator_across_examples_in_a_class,
                                                                                 name_accumulator_across_classes = name_accumulator_across_classes,
                                                                                 sample_size_per_class = sample_size_per_class,
                                                                                 order = order)

        if return_scores_after_class_reduction:
            return id_scores_after_class_reduction, ood_scores_after_class_reduction

        else:
            return id_scores, ood_scores
