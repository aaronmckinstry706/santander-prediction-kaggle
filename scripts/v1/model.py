import logging
from typing import Tuple


import numpy.random
import pandas
import torch
import torch.cuda as cuda
import torch.nn as nn
import torch.optim as optim


LOGGER = logging.getLogger(__name__)
COMPUTE_DEVICE = torch.device('cuda' if cuda.is_available() else 'cpu')
CPU_DEVICE = torch.device('cpu')


class LinearNN(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear_layer = nn.Linear(in_features, out_features)

    def forward(self, x) -> torch.Tensor:
        return self.linear_layer(x)

    def predict(self, unlabeled_data: pandas.DataFrame) -> pandas.Series:
        data_tensor: torch.Tensor = torch.from_numpy(unlabeled_data.values).float()
        data_tensor = data_tensor.to(COMPUTE_DEVICE)
        predictions: torch.Tensor = self.forward(data_tensor).detach().to(CPU_DEVICE)
        predictions_dataframe = pandas.DataFrame(data=predictions.numpy(), columns=['target'],
                                                 index=unlabeled_data.index)
        return predictions_dataframe['target']


def test_LinearNN():
    x = LinearNN(5, 1).to(COMPUTE_DEVICE)
    input_data = pandas.DataFrame(data=numpy.random.randint(0, 20, size=(3, 5)), columns=list('ABCDE'))
    input_data.set_index('A')
    output = x.predict(input_data)
    assert type(output) == pandas.Series
    assert output.shape[0] == 3
    assert output.index.equals(input_data.index)


def train_early_stopping(training_data: pandas.DataFrame, validation_data: pandas.DataFrame, steps=10, lr=0.0001) \
        -> Tuple[LinearNN, int]:
    training_data_tensor = torch.from_numpy(training_data.drop(columns='target').values).float()
    training_data_tensor = training_data_tensor.to(COMPUTE_DEVICE)
    training_label_tensor = torch.from_numpy(training_data[['target']].values).float()
    training_label_tensor = training_label_tensor.to(COMPUTE_DEVICE)

    validation_data_tensor = torch.from_numpy(validation_data.drop(columns='target').values).float()
    validation_data_tensor = validation_data_tensor.to(COMPUTE_DEVICE)
    validation_label_tensor = torch.from_numpy(validation_data[['target']].values).float()
    validation_label_tensor = validation_label_tensor.to(COMPUTE_DEVICE)

    linear_nn = LinearNN(training_data.shape[1] - 1, 1).to(COMPUTE_DEVICE)
    mse_loss = nn.MSELoss()

    def rmse(t1, t2) -> torch.Tensor:
        return torch.sqrt(mse_loss(t1, t2))

    def validation_loss():
        linear_nn.eval()
        loss = rmse(linear_nn(validation_data_tensor), validation_label_tensor)
        linear_nn.train()
        return loss

    optimizer = optim.LBFGS(linear_nn.parameters(), lr=lr)

    def evaluate_loss_closure():
        optimizer.zero_grad()
        loss: torch.Tensor = rmse(linear_nn(training_data_tensor), training_label_tensor)
        loss.backward()
        return loss

    best_nn = LinearNN(training_data.shape[1] - 1, 1).to(COMPUTE_DEVICE)
    best_nn.load_state_dict(linear_nn.state_dict())
    best_validation_loss = validation_loss()
    count = 0
    for _ in range(steps):
        optimizer.step(evaluate_loss_closure)
        current_validation_loss = validation_loss()
        if current_validation_loss < best_validation_loss:
            best_validation_loss = current_validation_loss
            best_nn.load_state_dict(linear_nn.state_dict())
            count += 1
        else:
            break

    return best_nn, count


def test_train_early_stopping():
    training_data = pandas.DataFrame(data=numpy.random.rand(20, 27),
                                     columns=list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') + ['target'])
    training_data.set_index('A')
    validation_data = pandas.DataFrame(data=numpy.random.rand(20, 27),
                                       columns=list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') + ['target'])
    validation_data.set_index('A')
    linear_nn, iterations = train_early_stopping(training_data, validation_data, steps=1000)
    assert type(linear_nn) == LinearNN
    assert type(iterations) == int


def train(training_data: pandas.DataFrame, steps=10, lr=0.0001) -> LinearNN:
    data_tensor = torch.from_numpy(training_data.drop(columns='target').values).float()
    data_tensor = data_tensor.to(COMPUTE_DEVICE)
    label_tensor = torch.from_numpy(training_data[['target']].values).float()
    label_tensor = label_tensor.to(COMPUTE_DEVICE)

    linear_nn = LinearNN(training_data.shape[1] - 1, 1).to(COMPUTE_DEVICE)
    mse_loss = nn.MSELoss()

    def rmse(t1, t2):
        return torch.sqrt(mse_loss(t1, t2))

    optimizer = optim.LBFGS(linear_nn.parameters(), lr=lr)

    def evaluate_loss_closure():
        optimizer.zero_grad()
        loss: torch.Tensor = rmse(linear_nn(data_tensor), label_tensor)
        loss.backward()
        return loss

    for i in range(steps):
        optimizer.step(evaluate_loss_closure)

    return linear_nn


def test_train():
    training_data = pandas.DataFrame(data=numpy.random.rand(20, 27),
                                     columns=list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') + ['target'])
    training_data.set_index('A')
    linear_nn = train(training_data)
    assert type(linear_nn) == LinearNN
