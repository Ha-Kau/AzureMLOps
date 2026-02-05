import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

def main():
    # 1. Start MLflow autologging (This is the "magic" part for Azure ML)
    mlflow.sklearn.autolog()

    # 2. Load and Prepare Data
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Model Training
    # We wrap this in a 'with' block for clean logging
    with mlflow.start_run():
        print("Training RandomForest model...")
        
        # You can change these hyperparameters to see different results in GitHub/Azure
        model = RandomForestClassifier(n_estimators=100, max_depth=3)
        model.fit(X_train, y_train)

        # 4. Evaluation
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log manual metrics if you want them to stand out in the Azure UI
        mlflow.log_metric("accuracy", accuracy)
        print(f"Model accuracy: {accuracy:.4f}")

        # 5. Register the model
        # This makes the model appear in the 'Models' tab of Azure ML Studio
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="iris_model",
            registered_model_name="iris_github_model"
        )

if __name__ == "__main__":
    main()
