from itertools import combinations

import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

from aima.learning import NaiveBayesLearner, DataSet
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

class NaiveBayesClassifier:

    def __init__(self, data, target):
        self.data = data
        self.target = target
        self.features = [col for col in data.columns if col != target]
        self.classes = data[target].unique()

    def get_class_data(self, cls):
        return self.data[self.data[self.target] == cls]

    def prior_probabilities(self):
        """
        Calculate prior probabilities P(C) for every class.
        """
        priors = {}
        total_count = len(self.data)
        class_counts = self.data[self.target].value_counts()

        for cls, count in class_counts.items():
            priors[cls] = count / total_count

        return priors

    def evidence_probability(self):
        """
        Estimate the probability of the evidence P(X) in the dataset:
        P(X) = 1 / (number of possible combinations of feature values)
        """
        total = 1
        for feature in self.features:
            total *= self.data[feature].nunique()
        return 1 / total

    def print_prior_probabilities(self, priors):
        rows = [{'Class': cls, 'Prior Probability': prob} for cls, prob in priors.items()]
        df = pd.DataFrame(rows)
        print("\nPrior Probabilities (P(C)):")
        print(df.to_string(index=False))

    def print_likelihoods(self, likelihoods):

        print("\nLikelihoods (P(X|C)):")

        for cls in self.classes:
            rows = []
            for feature in self.features:
                likelihood = likelihoods[cls][feature]
                for value, prob in likelihood.items():
                    rows.append({
                        'Class': cls,
                        'Feature': feature,
                        'Feature Value': value,
                        'Likelihood': prob
                    })

            # Create a DataFrame and display it
            df = pd.DataFrame(rows)
            print(f"\nClass {cls} Likelihoods:")
            print(df.to_string(index=False))

class MultinomialNaiveBayesClassifier(NaiveBayesClassifier):

    def likelihoods(self):
        """
        Calculate likelihoods P(X|C) for each class and feature.
        """
        likelihoods = {}
        for cls in self.classes:
            likelihoods[cls] = self.class_likelihood(cls)

        return likelihoods

    def class_likelihood(self, cls):
        class_data = self.get_class_data(cls)
        class_count = len(class_data)

        class_likelihoods = {}
        for feature in self.features:
            feature_count = class_data[feature].value_counts()
            class_likelihoods[feature] = self.feature_likelihood(feature, class_count, feature_count)

        return class_likelihoods

    def feature_likelihood(self, feature, class_count, feature_count):

        unique_values = self.data[feature].nunique()
        feature_likelihood = {}
        for value in self.data[feature].unique():
            count = feature_count.get(value, 0)
            # Laplace smoothing
            probability = (count + 1) / (class_count + unique_values)
            feature_likelihood[value] = probability

        return feature_likelihood

class GaussianNaiveBayesClassifier(NaiveBayesClassifier):

    def statistics(self):
        """
        Sets the mean and standard deviation for all features
        """
        stats = {}
        for cls in self.classes:
            class_data = self.get_class_data(cls)
            stats[cls] = self.class_statistics(class_data)
        return stats

    def class_statistics(self, class_data):
        stats = {}
        for feature in self.features:
            stats[feature] = self.feature_statistics(class_data, feature)
        return stats

    def feature_statistics(self, class_data, feature):
        """
        Returns the mean and standard deviation for a given feature.
        """
        return {
            'mean': class_data[feature].mean(),
            'std': class_data[feature].std()
        }

    def print_likelihoods(self, statistics):
        print("\nClass-Specific Statistics (Mean and Std by Feature):")

        for cls, features in statistics.items():
            rows = []
            for feature, stats in features.items():
                rows.append({
                    'Class': cls,
                    'Feature': feature,
                    'Mean': stats['mean'],
                    'Standard Deviation': stats['std']
                })

            df = pd.DataFrame(rows)
            print(f"\nClass {cls} Statistics:")
            print(df.to_string(index=False))

def calculate_probabilities_for_mushroom_data(mushroom_data):
    # Create the classifier
    mushroom_classifier = MultinomialNaiveBayesClassifier(mushroom_data, target="class")

    # Calculate the prior probabilities
    priors = mushroom_classifier.prior_probabilities()
    mushroom_classifier.print_prior_probabilities(priors)

    # Estimate the probability of the evidence
    evidence_probability = mushroom_classifier.evidence_probability()
    print(f"\nEstimated Probability of the Evidence (P(X)):")
    print(evidence_probability)

    # Determine the likelihood of the evidence (the numerator of Bayes’ formula)
    likelihoods = mushroom_classifier.likelihoods()
    mushroom_classifier.print_likelihoods(likelihoods)

def calculate_probabilities_for_banknote_data(banknote_data):

    banknote_classifier = GaussianNaiveBayesClassifier(banknote_data, target="class")

    # Calculate and print prior probabilities
    priors = banknote_classifier.prior_probabilities()
    banknote_classifier.print_prior_probabilities(priors)

    # Estimate the probability of the evidence
    evidence_prob = banknote_classifier.evidence_probability()
    print(f"\nEstimated Probability of the Evidence (P(X)):\n{evidence_prob:.6e}")

    # Calculate and print statistics (mean and standard deviation for each feature and class)
    statistics = banknote_classifier.statistics()
    banknote_classifier.print_likelihoods(statistics)

def load_mushroom_data():
    col_names = ['class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
                 'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
                 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
                 'stalk-surface-below-ring', 'stalk-color-above-ring',
                 'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
                 'ring-type', 'spore-print-color', 'population', 'habitat']

    # Load the dataset
    mushroom_data_file = "data/mushroom/agaricus-lepiota.data"
    mushroom_data = pd.read_csv(mushroom_data_file, header=None, names=col_names)

    # Encode all columns (including the target 'class')
    for col in mushroom_data.columns:
        mushroom_data[col] = mushroom_data[col].astype('category').cat.codes

    return mushroom_data

def load_banknote_data():
    data_file = "data/data_banknote_authentication.txt"
    columns = ["variance", "skewness", "curtosis", "entropy", "class"]
    banknote_data = pd.read_csv(data_file, header=None, names=columns)

    # Standardize the continuous features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(banknote_data.drop(columns=["class"]))
    scaled_data = pd.DataFrame(scaled_features, columns=columns[:-1])
    scaled_data["class"] = banknote_data["class"]

    return scaled_data

def prepare_data(data, target_col='class'):
    X = data.drop(target_col, axis=1)
    y = data[target_col]

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test

def create_dataset(X_train, y_train, data):
    # Prepare inputs and target indices
    features = X_train.columns.tolist()
    inputs = list(range(len(features)))
    target = len(features)

    # Create examples
    examples = X_train.values.tolist()

    # Append target
    for i, row in enumerate(examples):
        row.append(y_train.values[i])

    # Create the values dictionary
    values = {
        idx: list(data[col_name].unique())
        for idx, col_name in enumerate(features)
    }

    # Add target values
    values[target] = list(data['class'].unique())

    return DataSet(name='Data', examples=examples, inputs=inputs, target=target, values=values)

def evaluate_model(model, X_test, y_test):
    # Prepare examples
    examples = X_test.values.tolist()

    y_pred = []
    for row in examples:
        y_pred.append(model(row))  # Predict using the model

    # Evaluate the classifier
    correct = sum(1 for i in range(len(y_test)) if y_pred[i] == y_test.values[i])

    # Compute accuracy
    accuracy = correct / len(y_test)

    return accuracy, y_test.values, y_pred

def show_confusion_matrix(y_true, y_pred, class_names, dataset_name="Dataset"):

    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot using seaborn heatmap
    sns.set(font_scale=1.2)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)

    plt.title(f"Confusion Matrix - {dataset_name}")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

def decision_boundaries_for_all_pairs(model, X, y, feature_names, resolution=0.1):
    """
    This function was taken from the code provided by Ruairí D. O’Reilly's solution for lab 9
    Plot decision boundaries for all combinations of two features.
    """
    feature_combinations = list(combinations(range(X.shape[1]), 2))  # All 2-feature pairs
    for feature1, feature2 in feature_combinations:
        # Create grid for visualization
        x_min, x_max = X[:, feature1].min() - 1, X[:, feature1].max() + 1
        y_min, y_max = X[:, feature2].min() - 1, X[:, feature2].max() + 1
        xx1, xx2 = np.meshgrid(np.arange(x_min, x_max, resolution),
                               np.arange(y_min, y_max, resolution))

        # Fill grid data with predictions from the model
        grid_data = np.zeros((xx1.size, X.shape[1]))
        grid_data[:, feature1] = xx1.ravel()
        grid_data[:, feature2] = xx2.ravel()
        Z = np.array([model(row) for row in grid_data])
        Z = Z.reshape(xx1.shape)

        # Plot decision boundary
        plt.figure(figsize=(6, 6))
        plt.contourf(xx1, xx2, Z, alpha=0.8, cmap=ListedColormap(['red', 'blue', 'green']))
        plt.scatter(X[:, feature1], X[:, feature2], c=y, edgecolor='k', cmap=ListedColormap(['red', 'blue', 'green']))
        plt.xlabel(feature_names[feature1])
        plt.ylabel(feature_names[feature2])
        plt.title(f"Decision Boundary: {feature_names[feature1]} vs {feature_names[feature2]}")
        plt.show()

def train_data(data, dataset_name, continuous=False):
    """
    Train a Naive Bayes classifier and display visualizations for a dataset.

    Args:
        data (pd.DataFrame): The dataset to train on.
        dataset_name (str): Name of the dataset (used for labeling outputs).
        continuous (bool): Whether the dataset has continuous features.
    """
    # Step 1: Prepare the data
    X_train, X_test, y_train, y_test = prepare_data(data)

    # Step 2: Create Dataset object
    dataset = create_dataset(X_train, y_train, data)

    # Step 3: Train the Naive Bayes classifier
    model = NaiveBayesLearner(dataset, continuous=continuous)

    # Step 4: Evaluate the classifier
    accuracy, y_true, y_pred = evaluate_model(model, X_test, y_test)

    print(f"Accuracy for {dataset_name}: {accuracy * 100:.2f}%")

    # Step 5: Plot Confusion Matrix
    class_names = [str(cls) for cls in np.unique(y_true)]
    # Remove plt.title from here
    show_confusion_matrix(y_true, y_pred, class_names=class_names, dataset_name=dataset_name)

    # Step 6: Plot Decision Boundaries (if applicable)
    if continuous:
        X_test_np = X_test.values
        y_test_np = y_test.values
        feature_names = X_test.columns.tolist()
        print(f"Plotting decision boundaries for {dataset_name}...")
        decision_boundaries_for_all_pairs(model, X_test_np, y_test_np, feature_names)


mushroom_data = load_mushroom_data()
calculate_probabilities_for_mushroom_data(mushroom_data)
train_data(mushroom_data, dataset_name="Mushroom Dataset", continuous=False)

banknote_data = load_banknote_data()
calculate_probabilities_for_banknote_data(banknote_data)
train_data(banknote_data, dataset_name="Banknote Dataset", continuous=True)












