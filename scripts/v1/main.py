import argparse
import logging
import math
import pathlib
from typing import Tuple


import numpy
import pandas
import sklearn.metrics as metrics
import torch
import torch.cuda as cuda
import tqdm


from scripts.v1 import model
from scripts.v1 import xval


LOGGER = logging.getLogger(__name__)
COMPUTE_DEVICE = torch.device('cuda' if cuda.is_available() else 'cpu')
CPU_DEVICE = torch.device('cpu')


def run(clargs):
    def logify(x: pandas.Series):
        return numpy.log(x + 1.0)

    def delogify(x: pandas.Series):
        return numpy.exp(x) - 1.0

    LOGGER.info('Loading training data.')
    training_data = pandas.read_csv(str(clargs.training_data))
    training_data.set_index('ID', inplace=True)
    training_data = logify(training_data)

    def rmse(targets: pandas.Series, predictions: pandas.Series) -> float:
        return math.sqrt(metrics.mean_squared_error(targets, predictions))

    LOGGER.info('Choosing hyperparameters.')
    avg_xval_losses = []
    parameter_dicts = []
    NUM_LEARNING_RATES = 7
    MAX_STEPS = 100
    with tqdm.tqdm(total=NUM_LEARNING_RATES, position=0, desc='hyperparam sets') as lr_power_bar:
        for lr_power in range(0, NUM_LEARNING_RATES):
            lr = 0.1 ** lr_power
            parameter_dict = {'lr': lr, 'steps': MAX_STEPS}

            def validate(training_data: pandas.DataFrame, validation_data: pandas.DataFrame) -> Tuple[float, int]:
                linear_nn, count = model.train_early_stopping(training_data, validation_data, **parameter_dict)
                return rmse(validation_data['target'], linear_nn.predict(validation_data.drop(columns='target'))), count

            avg_xval_loss, avg_iteration = xval.cross_validate(training_data, validate, num_folds=10)

            parameter_dict['steps'] = int(avg_iteration)
            parameter_dicts.append(parameter_dict)
            avg_xval_losses.append(avg_xval_loss)

            lr_power_bar.update(1)

    min_index = avg_xval_losses.index(min(avg_xval_losses))
    best_params = parameter_dicts[min_index]
    LOGGER.info('Choosing {}.'.format(best_params))

    LOGGER.info('Fitting model.')
    linear_nn = model.train(training_data, **best_params)
    log_prediction = linear_nn.predict(training_data.drop(columns='target'))
    rmsle = rmse(training_data['target'], log_prediction)
    LOGGER.info('Training RMSLE: {0}'.format(rmsle))

    LOGGER.info('Loading testing data.')
    testing_data = pandas.read_csv(str(clargs.testing_data))
    testing_data.set_index('ID', inplace=True)

    LOGGER.info('Making predictions.')
    output = pandas.DataFrame(index=testing_data.index)
    output['target'] = linear_nn.predict(testing_data)
    output['target'] = delogify(output['target'].clip(lower=0.0, upper=math.log(4600000000000.0 + 1.0)))

    clargs.output_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info('Outputting predictions to {}'.format(clargs.output_path))
    output.to_csv(str(clargs.output_path))


def add_args(arg_parser: argparse.ArgumentParser):
    arg_parser.description = ('LBFGS with root-mean-squared log error. No regularization. Default params '
                              'from `sklearn`. The targets used during regression were `log(a_i + 1)`. When '
                              'predicting, the regression model predicted these log values, and then exponentials '
                              '`exp(p_i) - 1` were taken before outputting the prediction file. To prevent inf/NaN '
                              'values, the prediction values `p_i` were clipped to values in the range [0, '
                              'log(4600000000000 + 1)] (the largest transaction ever made, according to a quick '
                              'Google search, was 4.6 trillion dollars). For preprocessing, the `log(x + 1)` transform '
                              'was applied to all numeric values. Early stopping was used for regularization during '
                              'cross-validation; cross-validation was used to determine the best learning rate ('
                              'values 1, 0.1, ..., 10^-6 were tested), and the average early-stop-time was used to'
                              'determine how long to train the model on the whole training set.')

    def nonexistent_file_path(s: str):
        try:
            p = pathlib.Path(s)
        except TypeError:
            raise argparse.ArgumentTypeError('{} is not a valid path.'.format(s))
        if p.exists():
            raise argparse.ArgumentTypeError('Path {} must not exist.'.format(s))
        return p

    def existent_file_path(s: str):
        try:
            p = pathlib.Path(s)
        except TypeError:
            raise argparse.ArgumentTypeError('{} is not a valid path.'.format(s))
        if not p.exists() or not p.is_file():
            raise argparse.ArgumentTypeError('Path {} must be a file.'.format(s))
        return p

    arg_parser.add_argument('--output-path', '-o', type=nonexistent_file_path,
                            default='predictions/v1.csv',
                            help='The directory to which you want the predictions output. Default is '
                                 '"predictions/v1.csv".')
    arg_parser.add_argument('--training-data', type=existent_file_path,
                            default='data/train.csv',
                            help='The path to the training data CSV file, downloaded from the Kaggle competition site. '
                                 'Default is "data/train.csv".')
    arg_parser.add_argument('--testing-data', type=existent_file_path,
                            default='data/test.csv',
                            help='The path to the testing data CSV file, downloaded from the Kaggle competition site. '
                                 'Default is "data/test.csv".')
