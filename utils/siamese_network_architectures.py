# Citation:
# 1. https://github.com/fangpin/siamese-pytorch/blob/master/model.py
# 2. https://github.com/ae-foster/pytorch-simclr/blob/dc9ac57a35aec5c7d7d5fe6dc070a975f493c1a5/critic.py#L5

import os
import torch
import numpy as np

# citation
# 1. https://github.com/ae-foster/pytorch-simclr/blob/dc9ac57a35aec5c7d7d5fe6dc070a975f493c1a5/critic.py#L5
class LinearCritic(torch.nn.Module):
    def __init__(self, latent_dim, projection_dim):
        super(LinearCritic, self).__init__()
        self.projection_dim = projection_dim
        self.w1 = torch.nn.Linear(in_features = latent_dim, out_features = latent_dim)
        self.bn1 = torch.nn.BatchNorm1d(num_features = latent_dim)
        self.relu = torch.nn.ReLU()
        self.w2 = torch.nn.Linear(in_features = latent_dim, out_features = self.projection_dim)
        self.bn2 = torch.nn.BatchNorm1d(num_features = self.projection_dim)

    def forward(self, x):
        out = self.w1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.w2(out)
        out = self.bn2(out)

        return out

# Siamese Network class
# the siamese network class follows the implementation in https://github.com/fangpin/siamese-pytorch/blob/master/model.py
class SiameseNetworkSkeleton(torch.nn.Module):
    def __init__(self, shared_model):
        super(SiameseNetworkSkeleton, self).__init__()
        self.shared_model = shared_model

    # this is called forward_one in the github code we copied it from
    def get_embedding(self, x):
        embedding = self.shared_model.get_embedding(x)

        return embedding

    def distance(self, embedding_1, embedding_2):
        raise NotImplementedError("Please implement this method.")

    def forward(self, x1, x2):
        embedding_1 = self.get_embedding(x = x1)
        embedding_2 = self.get_embedding(x = x2)
        output = self.distance(embedding_1 = embedding_1, embedding_2 = embedding_2)

        return output

    def extract_last_layer_weights(self):
        # extracting weight
        weight = self.last_layer.weight.detach().cpu().numpy()
        assert len(weight.shape) == 2

        # extracting bias
        bias = None
        if self.last_layer.bias is not None:
            bias = np.reshape(a = self.last_layer.bias.detach().cpu().numpy(), newshape = (weight.shape[0], 1))
        else:
            bias = np.zeros(shape = (weight.shape[0], 1))

        # printing stuff
        print()
        print("Model last layer weight shape: ", weight.shape)
        print("Model last layer bias shape: ", bias.shape)
        print()

        # checking if everything is okay
        assert len(bias.shape) == 2
        assert bias.shape[0] == weight.shape[0]

        return weight, bias

# weighted l1
class SiameseNetworkVersion1(SiameseNetworkSkeleton):
    def __init__(self, shared_model):
        super(SiameseNetworkVersion1, self).__init__(shared_model = shared_model)
        self.projection_dim = self.shared_model.num_output_channels
        self.last_layer = torch.nn.Linear(in_features = self.projection_dim, out_features = 1, bias = True)

    def distance(self, embedding_1, embedding_2):
        assert embedding_1.shape == embedding_2.shape

        dist = torch.abs(embedding_1 - embedding_2)
        output = self.last_layer(dist)

        return output

# weighted l2
class SiameseNetworkVersion2(SiameseNetworkSkeleton):
    def __init__(self, shared_model):
        super(SiameseNetworkVersion2, self).__init__(shared_model = shared_model)
        self.projection_dim = self.shared_model.num_output_channels
        self.last_layer = torch.nn.Linear(in_features = self.projection_dim, out_features = 1, bias = True)

    def distance(self, embedding_1, embedding_2):
        assert embedding_1.shape == embedding_2.shape

        dist = embedding_1 - embedding_2
        output = torch.mul(dist, dist)
        output = self.last_layer(output)

        return output

# M score
class SiameseNetworkVersion3(SiameseNetworkSkeleton):
    def __init__(self, shared_model):
        super(SiameseNetworkVersion3, self).__init__(shared_model = shared_model)
        self.projection_dim = self.shared_model.num_output_channels
        self.last_layer = torch.nn.Linear(in_features = self.projection_dim, out_features = self.projection_dim, bias = False)

    def distance(self, embedding_1, embedding_2):
        assert embedding_1.shape == embedding_2.shape

        dist = embedding_1 - embedding_2
        M_dist = self.last_layer(dist)
        mult_matrix = torch.mul(dist, M_dist)
        output = torch.sum(mult_matrix, dim = 1, keepdim = False)

        return output

# cosine similarity
class SiameseNetworkVersion4(SiameseNetworkSkeleton):
    def __init__(self, shared_model, projection_dim, temp = 0.1):
        super(SiameseNetworkVersion4, self).__init__(shared_model = shared_model)
        self.projection_dim = projection_dim
        self.linear_layer = LinearCritic(latent_dim = self.shared_model.num_output_channels, projection_dim = self.projection_dim)
        self.last_layer = torch.nn.CosineSimilarity(dim = -1)

        assert temp is not None
        self.temp = float(temp)
        assert self.temp > 0

    # -1.0 comes because of our choice of pairwise label
    # label 0 -> images of the pair are from the same class
    # label 1 -> images of the pair are from different classes
    def distance(self, embedding_1, embedding_2):
        output = (-1.0) * self.last_layer(embedding_1, embedding_2) / self.temp
        return output

    def extract_last_layer_weights(self):
        raise ValueError("There is no weight for last layer, last layer is cosine similarity.")

# uses non-weighted L2 distance (or simple Euclidean distance)
class SiameseNetworkVersion5(SiameseNetworkSkeleton):
    def __init__(self, shared_model):
        super(SiameseNetworkVersion5, self).__init__(shared_model = shared_model)
        self.projection_dim = self.shared_model.num_output_channels

    def distance(self, embedding_1, embedding_2):
        assert embedding_1.shape == embedding_2.shape

        diff = embedding_1 - embedding_2
        output = torch.mul(diff, diff)
        output = torch.sum(output, dim = 1, keepdim = False)

        return output

    def extract_last_layer_weights(self):
        raise ValueError("There is no weight for last layer, last layer is simple Euclidean distance.")


def print_message(message):
    print()
    print(message)
    print()

# summary method
def createSiameseNetwork(shared_model, siamese_network_version, temp = 0.1, projection_dim = 128):
    if siamese_network_version == 1:
        print_message(message = "Siamese network that uses weighted L1 distance.")
        model = SiameseNetworkVersion1(shared_model = shared_model)

    elif siamese_network_version == 2:
        print_message(message = "Siamese network that uses weighted L2 distance.")
        model = SiameseNetworkVersion2(shared_model = shared_model)

    elif siamese_network_version == 3:
        print_message(message = "Siamese network that uses M distance.")
        model = SiameseNetworkVersion3(shared_model = shared_model)

    elif siamese_network_version == 4:
        print_message(message = "Siamese network that uses cosine similarity.")
        model = SiameseNetworkVersion4(shared_model = shared_model, projection_dim = projection_dim, temp = temp)

    elif siamese_network_version == 5:
        print_message(message = "Siamese network that uses an MSP model, with Euclidean distance at the end.")
        model = SiameseNetworkVersion5(shared_model = shared_model)

    else:
        raise ValueError("Given siamese network version not implemented.")

    return model
