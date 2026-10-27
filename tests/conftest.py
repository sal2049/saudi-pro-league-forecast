import pytest

from spl_predictor.data import load_matches
from spl_predictor.features import add_features


@pytest.fixture(scope="session")
def matches():
    return load_matches()


@pytest.fixture(scope="session")
def data(matches):
    return add_features(matches)
