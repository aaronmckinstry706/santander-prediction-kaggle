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
    training_data = pandas.read_csv('data/train.csv')
    training_data['log_target'] = numpy.log(training_data['target'] + 1.0)

    LOGGER.info('Fitting model.')
    linear_regression = linear_model.LinearRegression()
    linear_regression.fit(training_data.drop(columns=['log_target', 'target', 'ID']), training_data['log_target'])
    training_data['log_prediction'] = linear_regression.predict(training_data.drop(columns=['log_target', 'target', 'ID']))
    LOGGER.info('Training RMSLE: {0}'.format(math.sqrt(metrics.mean_squared_error(
        training_data['log_target'], training_data['log_prediction']))))

    LOGGER.info('Loading testing data.')
    testing_data = pandas.read_csv('data/test.csv')

    LOGGER.info('Making predictions.')
    log_targets = linear_regression.predict(testing_data.drop(columns=['ID']))
    testing_data['log_target'] = log_targets
    testing_data['log_target'] = testing_data['log_target'].clip(lower=0.0, upper=math.log(4600000000000.0 + 1.0))
    testing_data['target'] = numpy.exp(testing_data['log_target']) - 1.0

    output_dir = pathlib.Path('data')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'baseline_linear_regression.csv'
    LOGGER.info('Outputting predictions to {}'.format(output_path))
    testing_data[['ID', 'target']].to_csv(str(output_path), index=False)


def add_args(arg_parser: argparse.ArgumentParser):
    arg_parser.description = ('Standard linear regression, with mean squared error. No regularization. Default params '
                              'from `sklearn`. The targets used during regression were `log(a_i + 1)`. When '
                              'predicting, the regression model predicted these log values, and then exponentials '
                              '`exp(p_i) - 1` were taken before outputting the prediction file. To prevent inf/NaN '
                              'values, the prediction values `p_i` were clipped to values in the range [0, '
                              'log(4600000000000 + 1)] (the largest transaction ever made, according to a quick '
                              'Google search, was 4.6 trillion dollars).')

    def dir_path_arg(s: str):
        p = pathlib.Path(s)
        if not p.is_dir():
            raise argparse.ArgumentTypeError('Path {} is not a directory.'.format(p))
        return p

    arg_parser.add_argument('--output-dir', '-o', type=dir_path_arg, default='data',
                            help='The directory to which you want the baseline predictions output. Default is "data".')
