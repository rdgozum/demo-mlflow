"""Main module."""
import os
import sys

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.abspath(".."))

from demo_mlflow.component.tracking.tracking_demo import TrackingDemo

tracker = TrackingDemo()
tracker.initialize_experiment()


def log():
    print("Start logging...")

    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    for idx, depth in enumerate([1, 2, 5, 10, 20]):
        clf = DecisionTreeClassifier(max_depth=depth)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # MLFlow Parameters
        items_dict = {
            "parameters": {"depth": depth},
            "metrics": {"accuracy": accuracy},
            "models": {"clf": clf},
        }
        RUN_NAME = f"run_{idx}"

        tracker.log(run_name=RUN_NAME, **items_dict)


def query():
    print("Start querying...")

    tracker.query()


if __name__ == "__main__":
    log()
    query()
