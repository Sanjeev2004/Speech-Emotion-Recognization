"""Classical machine learning on the hand-crafted 'stats' vector (input: (N, D))."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def build_svm(seed: int = 42):
    # RBF-kernel SVM; features must be scaled first
    return make_pipeline(StandardScaler(),
                         SVC(C=10, gamma="scale", class_weight="balanced",
                             random_state=seed))


def build_random_forest(seed: int = 42):
    return RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced",
                                  n_jobs=-1, random_state=seed)
