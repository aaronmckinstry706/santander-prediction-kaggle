import argparse
import logging


LOGGER = logging.getLogger(__name__)


def run(clargs):
    LOGGER.debug('Running sample_package_script with arguments: {}.'.format(clargs))


def add_args(arg_parser: argparse.ArgumentParser):
    LOGGER.debug('Adding arguments for sample_package_script.')
    arg_parser.description = 'This is a sample script; it is a package.'
    arg_parser.add_argument('--sample-arg', default='sample-value', type=str, help='This is a sample argument.')
