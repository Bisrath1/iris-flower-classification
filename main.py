import pandas as pd
from src.preprocess import split_features_labels, train_test_data, scale_features
from src.model import train_knn, evaluate_model

def main():
    # Load dataset
    df = pd.read_csv("data/IRIS.csv")

    # Preprocess
    X, y = split_features_labels(df)
    X_train, X_test, y_train, y_test = train_test_data(X, y)
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    # Train model
    model = train_knn(X_train_scaled, y_train, neighbors=3)

    # Evaluate
    accuracy = evaluate_model(model, X_test_scaled, y_test)
    print(f"✅ Model Accuracy: {accuracy:.2f}")

    # Sample prediction
    sample = [[5, 2.9, 1, 0.2]]
    pred = model.predict(sample)
    print(f"🌸 Prediction for {sample}: {pred[0]}")

if __name__ == "__main__":
    main()
