# iris_classification.py
# Iris Flower Classification using Machine Learning
# Author: Chandana H G

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_data():
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name="species")
    return X, y, iris.target_names

def visualize_data(X, y, target_names):
    y_named = y.map({i: name for i, name in enumerate(target_names)})
    df = pd.concat([X, y_named.rename("species")], axis=1)
    sns.pairplot(df, hue="species")
    plt.suptitle("Pairplot of Iris Dataset", y=1.02)
    plt.show()

def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, target_names):
    y_pred = model.predict(X_test)
    print("🔍 Model Evaluation")
    print("-------------------")
    print("✅ Accuracy:", accuracy_score(y_test, y_pred))
    print("\n📊 Classification Report:\n", classification_report(y_test, y_pred, target_names=target_names))
    print("🧮 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

def predict_sample(model, target_names):
    sample = [[5.1, 3.5, 1.4, 0.2]]
    predicted = model.predict(sample)[0]
    print("\n🌼 Prediction for sample {}: {}".format(sample[0], target_names[predicted]))

def main():
    # Load and visualize data
    X, y, target_names = load_data()
    visualize_data(X, y, target_names)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    evaluate_model(model, X_test, y_test, target_names)

    # Predict custom sample
    predict_sample(model, target_names)

if __name__ == "__main__":
    main()
