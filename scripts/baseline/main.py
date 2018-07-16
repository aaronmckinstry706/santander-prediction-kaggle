import argparse
import logging
import math
import pathlib


import numpy
import pandas
import sklearn.linear_model as linear_model
import sklearn.metrics as metrics


LOGGER = logging.getLogger(__name__)


def run(clargs):
    LOGGER.info('Loading training data.')
    training_data = pandas.read_csv(str(clargs.training_data))
    training_data['log_target'] = numpy.log(training_data['target'] + 1.0)

    LOGGER.info('Fitting model.')
    linear_regression = linear_model.LinearRegression()
    linear_regression.fit(training_data.drop(columns=['log_target', 'target', 'ID']), training_data['log_target'])
    training_data['log_prediction'] = linear_regression.predict(
        training_data.drop(columns=['log_target', 'target', 'ID']))
    rmsle = math.sqrt(metrics.mean_squared_error(training_data['log_target'], training_data['log_prediction']))
    LOGGER.info('Training RMSLE: {0}'.format(rmsle))

    LOGGER.info('Loading testing data.')
    testing_data = pandas.read_csv(str(clargs.testing_data))

    LOGGER.info('Making predictions.')
    log_targets = linear_regression.predict(testing_data.drop(columns=['ID']))
    testing_data['log_target'] = log_targets
    testing_data['log_target'] = testing_data['log_target'].clip(lower=0.0, upper=math.log(4600000000000.0 + 1.0))
    testing_data['target'] = numpy.exp(testing_data['log_target']) - 1.0

    clargs.output_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info('Outputting predictions to {}'.format(clargs.output_path))
    testing_data[['ID', 'target']].to_csv(str(clargs.output_path), index=False)


def add_args(arg_parser: argparse.ArgumentParser):
    arg_parser.description = ('Standard linear regression, with mean squared error. No regularization. Default params '
                              'from `sklearn`. The targets used during regression were `log(a_i + 1)`. When '
                              'predicting, the regression model predicted these log values, and then exponentials '
                              '`exp(p_i) - 1` were taken before outputting the prediction file. To prevent inf/NaN '
                              'values, the prediction values `p_i` were clipped to values in the range [0, '
                              'log(4600000000000 + 1)] (the largest transaction ever made, according to a quick '
                              'Google search, was 4.6 trillion dollars).')

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
                            default='predictions/baseline_linear_regression.csv',
                            help='The directory to which you want the baseline predictions output. Default is '
                                 '"predictions/baseline_linear_regression.csv".')
    arg_parser.add_argument('--training-data', type=existent_file_path,
                            default='data/train.csv',
                            help='The path to the training data CSV file, downloaded from the Kaggle competition site. '
                                 'Default is "data/train.csv".')
    arg_parser.add_argument('--testing-data', type=existent_file_path,
                            default='data/test.csv',
                            help='The path to the testing data CSV file, downloaded from the Kaggle competition site. '
                                 'Default is "data/test.csv".')
