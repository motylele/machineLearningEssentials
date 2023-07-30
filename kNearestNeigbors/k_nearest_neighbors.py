from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import datasets


def load_iris_data():
    """Loading Iris dataset"""
    iris = datasets.load_iris()
    features = iris.data
    target = iris.target
    return features, target


def load_wine_data():
    """Loading Wine dataset"""
    wine = datasets.load_wine()
    features = wine.data
    target = wine.target
    return features, target


def preprocess_data(features):
    """Standardize the features"""
    standardizer = StandardScaler()
    features_std = standardizer.fit_transform(features)
    return features_std


def train_knn_classifier(features_train, target_train, n_neighbors=5, metric='minkowski'):
    """Training k Nearest Neighbor classifier"""
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric, n_jobs=-1)
    knn.fit(features_train, target_train)
    return knn


def evaluate_classifier(knn, features_test, target_test):
    """Classifier evaluation accuracy"""
    accuracy = knn.score(features_test, target_test)
    return accuracy


def main():
    # Load Iris dataset
    features_iris, target_iris = load_iris_data()
    features_train_iris, features_test_iris, target_train_iris, target_test_iris = train_test_split(
        features_iris, target_iris, test_size=0.2, random_state=42
    )

    # Preprocess Iris data
    features_train_iris_std = preprocess_data(features_train_iris)
    features_test_iris_std = preprocess_data(features_test_iris)

    # Train and evaluate KNN classifiers for the Iris dataset
    knn_iris = train_knn_classifier(features_train_iris_std, target_train_iris)
    accuracy_iris = evaluate_classifier(knn_iris, features_test_iris_std, target_test_iris)

    print("Iris dataset - KNN Classifier Accuracy: {:.2f}".format(accuracy_iris))

    # Make predictions on the Iris test data
    predicted_labels_iris = knn_iris.predict(features_test_iris_std)

    # Display the predictions and actual labels for Iris dataset
    print("\nPredicted Labels for Iris dataset:")
    print(predicted_labels_iris)
    print("Actual Labels for Iris dataset:")
    print(target_test_iris)

    # Load Wine dataset
    features_wine, target_wine = load_wine_data()
    features_train_wine, features_test_wine, target_train_wine, target_test_wine = train_test_split(
        features_wine, target_wine, test_size=0.2, random_state=42
    )

    # Preprocess Wine data
    features_train_wine_std = preprocess_data(features_train_wine)
    features_test_wine_std = preprocess_data(features_test_wine)

    # Train and evaluate KNN classifiers for the Wine dataset
    knn_wine = train_knn_classifier(features_train_wine_std, target_train_wine, n_neighbors=3, metric='euclidean')
    accuracy_wine = evaluate_classifier(knn_wine, features_test_wine_std, target_test_wine)

    print("\nWine dataset - KNN Classifier Accuracy: {:.2f}".format(accuracy_wine))

    # Make predictions on the Wine test data
    predicted_labels_wine = knn_wine.predict(features_test_wine_std)

    # Display the predictions and actual labels for Wine dataset
    print("\nPredicted Labels for Wine dataset:")
    print(predicted_labels_wine)
    print("Actual Labels for Wine dataset:")
    print(target_test_wine)


if __name__ == "__main__":
    main()
