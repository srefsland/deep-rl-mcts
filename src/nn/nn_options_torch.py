import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

activation_functions = {
    "relu": F.relu,
    "sigmoid": torch.sigmoid,
    "tanh": torch.tanh,
    "softmax": lambda input, dim=1: F.softmax(input, dim=dim),
    "linear": lambda x: x,
}

optimizers = {
    "Adagrad": optim.Adagrad,
    "SGD": optim.SGD,
    "RMSprop": optim.RMSprop,
    "Adam": optim.Adam,
}

loss_functions = {
    "categorical_crossentropy": nn.CrossEntropyLoss,
    "sparse_categorical_crossentropy": nn.CrossEntropyLoss,
    "kl_divergence": nn.KLDivLoss,
    "mse": nn.MSELoss,
}
