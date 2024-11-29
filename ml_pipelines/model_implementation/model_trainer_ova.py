import numpy as np
import pandas as pd
from logregression_utils import LogisticRegressionUtils
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, classification_report
import joblib

#Class that trains models with One Vs All Logistic Regression (Multi Class models)
class ModelTrainer:
    def __init__(self, train_data_path, test_data_path, model_output_path):
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.model_output_path = model_output_path
        self.models = {}  # Store one model (weights) per class

    def load_data(self):
        X_train = pd.read_csv(f"{self.train_data_path}/X_train.csv")
        X_test = pd.read_csv(f"{self.test_data_path}/X_test.csv")
        y_train = pd.read_csv(f"{self.train_data_path}/y_train.csv").values.flatten()  # Convert to 1D array
        y_test = pd.read_csv(f"{self.test_data_path}/y_test.csv").values.flatten()
        return X_train, X_test, y_train, y_test

    def train_one_vs_all(self, X_train, y_train):
        classes = np.unique(y_train)
        for c in classes:
            print(f"Training model for class {c}...")
            # Create binary labels for class c
            y_binary = (y_train == c).astype(int)
            initial_w = np.zeros(X_train.shape[1] + 1)

            # Optimize cost function
            res = minimize(
                fun=lambda w: LogisticRegressionUtils.costFunction(w, X_train.values, y_binary, lambda_=1.0),
                x0=initial_w,
                jac=True,
                method='TNC'
            )

            if res.success:
                print(f"Model for class {c} trained successfully.")
                self.models[c] = res.x
            else:
                raise ValueError(f"Optimization failed for class {c}.")

    def predict_one_vs_all(self, X_test):
        X_test = np.concatenate([np.ones((X_test.shape[0], 1)), X_test.values], axis=1)
        probabilities = {}
        for c, weights in self.models.items():
            probabilities[c] = LogisticRegressionUtils.sigmoid(X_test @ weights)
        # Assign the class with the highest probability
        predictions = np.array([max(probabilities, key=lambda c: probabilities[c][i]) for i in range(X_test.shape[0])])
        return predictions

    def evaluate(self, X_test, y_test):
        predictions = self.predict_one_vs_all(X_test)
        print("Classification Report:")
        print(classification_report(y_test, predictions))
        print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")

    def save_models(self):
        joblib.dump(self.models, self.model_output_path)
        print(f"Models saved to {self.model_output_path}")

    def run_pipeline(self):
        print("Starting model training pipeline...")
        X_train, X_test, y_train, y_test = self.load_data()
        print("Data loaded successfully.")
        
        print("Training one-vs-all models...")
        self.train_one_vs_all(X_train, y_train)
        
        print("Evaluating the model...")
        self.evaluate(X_test, y_test)
        
        print("Saving the models...")
        self.save_models()
        print("Pipeline completed successfully.")
