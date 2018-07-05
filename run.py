import argparse
import importlib
import logging
import sys


import script_utils as utils


LOGGER = logging.getLogger(__name__)


def add_args(arg_parser):
    subparsers = arg_parser.add_subparsers()
    for script_name in utils.script_names():
        script = importlib.import_module('scripts.{}'.format(script_name))
        script_subparser = subparsers.add_parser(name=script_name)
        script.add_args(script_subparser)
        script_subparser.set_defaults(func=script.run)


def main(raw_args):
    logging.basicConfig(format='%(asctime)s %(levelname)s [%(module)s.%(funcName)s:%(lineno)d] %(message)s',
                        stream=sys.stderr, level=logging.DEBUG)

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-v', action='count', default=0,
                            help='Determines verbosity of output. Include once to output basic information (INFO), '
                                 'twice to output debugging information (DEBUG).')
    add_args(arg_parser)
    clargs = arg_parser.parse_args(raw_args)

    root_logger = logging.getLogger()
    if clargs.v == 1:
        root_logger.setLevel(logging.INFO)
    elif clargs.v > 1:
        root_logger.setLevel(logging.DEBUG)
    else:
        root_logger.setLevel(logging.WARN)

    clargs.func(clargs)


if __name__ == '__main__':
    main(sys.argv[1:])
