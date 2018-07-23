import argparse
import logging
import math
import pathlib


import numpy
import pandas
import sklearn.linear_model as linear_model
import sklearn.metrics as metrics


from scripts.baseline import xval


LOGGER = logging.getLogger(__name__)


def run(clargs):
    def logify(x: pandas.Series):
        return numpy.log(x + 1.0)

    def delogify(x: pandas.Series):
        return numpy.exp(x) - 1.0

    LOGGER.info('Loading training data.')
    training_data = pandas.read_csv(str(clargs.training_data))
    training_data.set_index('ID', inplace=True)
    training_data['target'] = logify(training_data['target'])

    def train(training_data: pandas.DataFrame) -> linear_model.LinearRegression:
        linear_regression = linear_model.LinearRegression()
        linear_regression.fit(training_data.drop(columns='target'), training_data['target'])
        return linear_regression

    def rmse(targets: pandas.Series, predictions: pandas.Series) -> float:
        return math.sqrt(metrics.mean_squared_error(targets, predictions))

    def validate(training_data: pandas.DataFrame, validation_data: pandas.DataFrame) -> float:
        linear_regression = train(training_data)
        return rmse(validation_data['target'], linear_regression.predict(validation_data.drop(columns='target')))

    LOGGER.info('Cross validating.')
    xval_loss, xval_losses = xval.cross_validate(training_data, validate, num_folds=10)
    LOGGER.info('Cross-Validation RMSLE: {} = avg({})'.format(xval_loss, xval_losses))

    LOGGER.info('Fitting model.')
    linear_regression = train(training_data)
    log_prediction = linear_regression.predict(training_data.drop(columns='target'))
    rmsle = rmse(training_data['target'], log_prediction)
    LOGGER.info('Training RMSLE: {0}'.format(rmsle))

    LOGGER.info('Loading testing data.')
    testing_data = pandas.read_csv(str(clargs.testing_data))
    testing_data.set_index('ID', inplace=True)

    LOGGER.info('Making predictions.')
    output = pandas.DataFrame(index=testing_data.index)
    output['target'] = linear_regression.predict(testing_data)
    output['target'] = delogify(output['target'].clip(lower=0.0, upper=math.log(4600000000000.0 + 1.0)))

    clargs.output_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info('Outputting predictions to {}'.format(clargs.output_path))
    output.to_csv(str(clargs.output_path))


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
