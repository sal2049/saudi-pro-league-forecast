import pytest

from spl_predictor.data import load_matches, load_upcoming_teams
from spl_predictor.features import build_dataset


@pytest.fixture(scope="session")
def matches():
    return load_matches()


@pytest.fixture(scope="session")
def dataset(matches):
    return build_dataset(matches, load_upcoming_teams("2025-26"))
