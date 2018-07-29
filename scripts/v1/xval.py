import logging
from typing import Callable, List, Tuple


import numpy
import pandas
from tqdm import tqdm


LOGGER = logging.getLogger(__name__)


def folds(training_data: pandas.DataFrame, num_folds: int):
    rows, cols = training_data.shape
    fold_indexes = [i * len(training_data) // num_folds for i in range(num_folds + 1)]
    for i in range(num_folds):
        validation_subset = training_data.iloc[fold_indexes[i]:fold_indexes[i + 1]]
        training_subset = pandas.concat([
            training_data.iloc[:fold_indexes[i]],
            training_data.iloc[fold_indexes[i + 1]:]
        ])
        assert len(training_subset) + len(validation_subset) == rows, (
            'training size {}, validation size {} should add to total size {}'.format(
                len(training_subset), len(validation_subset), len(training_data)))
        yield training_subset, validation_subset


def test_folds():
    training_data = pandas.DataFrame(data=numpy.random.rand(4, 4), columns=list('ABCD'))
    num_folds = 3
    actual_folds = [t for t in folds(training_data, num_folds)]
    expected_folds = [
        # Expected validation folds are 0:1, 1:2, and 2:4.
        (pandas.concat([training_data.iloc[:0], training_data.iloc[1:]]), training_data.iloc[0:1]),
        (pandas.concat([training_data.iloc[:1], training_data.iloc[2:]]), training_data.iloc[1:2]),
        (pandas.concat([training_data.iloc[:2], training_data.iloc[4:]]), training_data.iloc[2:4])
    ]
    assert num_folds == len(actual_folds), 'expected {} folds, but found {} folds.'.format(num_folds, len(actual_folds))
    for ((expected_train, expected_val), (actual_train, actual_val)) in zip(expected_folds, actual_folds):
        assert expected_train.equals(actual_train), (
            'From training data:\n{}\nexpected training subset:\n{}\nactual training subset:\n{}\n'.format(
                training_data, expected_train, actual_train))
        assert expected_val.equals(actual_val), (
            'From training data:\n{}\nexpected validation subset:\n{}\nactual validation subset:\n{}\n'.format(
                training_data, expected_val, actual_val))


def cross_validate(training_data: pandas.DataFrame,
                   validation_score: Callable[[pandas.DataFrame, pandas.DataFrame], float],
                   num_folds: int) -> Tuple[float, List[float]]:
    scores = []
    for training_subset, validation_subset in folds(training_data, num_folds):
        scores.append(validation_score(training_subset, validation_subset))
    return sum(scores) / len(scores), scores


def test_cross_validate():
    def next_int(x: pandas.DataFrame, y: pandas.DataFrame) -> float:
        next_int.x = next_int.x + 1
        return next_int.x

    next_int.x = 0

    training_data = pandas.DataFrame(data=numpy.random.randn(4, 4), columns=['A', 'B', 'C', 'D'])
    num_folds = 3
    mean_score, all_scores = cross_validate(training_data, next_int, num_folds)
    assert tuple(all_scores) == (1, 2, 3), 'Scores should be (1, 2, 3), but are {}.'.format(all_scores)
    assert mean_score == 2, 'Mean score should be 2, but is {}.'.format(mean_score)
