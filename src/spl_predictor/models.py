"""The four candidates: a no-skill baseline, then simple to flexible.

Every candidate takes the gap feature of a match and returns three
probabilities: away win, draw, home win.
"""

from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

SEED = 42
BASELINE = "Always home win"


def candidates() -> dict:
    """Fresh, unfitted copies of every model in the comparison."""
    return {
        # Ignores the features: always picks the most common result (a home
        # win) and gives every match the training seasons' result frequencies.
        BASELINE: DummyClassifier(strategy="prior"),
        "Logistic regression": LogisticRegression(max_iter=1000),
        "Random forest": RandomForestClassifier(n_estimators=300, random_state=SEED),
        "Gradient boosting": GradientBoostingClassifier(random_state=SEED),
    }
