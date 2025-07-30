from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def train_knn(x_train, y_train, neighbors=1):
    """Train a KNN model with given neighbors."""
    model = KNeighborsClassifier(n_neighbors=neighbors)
    model.fit(x_train, y_train)
    return model

def evaluate_model(model, x_test, y_test):
    """Evaluate model accuracy."""
    preds = model.predict(x_test)
    return accuracy_score(y_test, preds)
