import numpy as np
import pandas as pd
from logregression_utils import LogisticRegressionUtils
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report



class ModelTrainer:

    #Initial method that runs as soon as class is instantiated
    def __init__(self, train_data_path, test_data_path, model_output_path):
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.model_output_path = model_output_path
        self.weights = None


    #Import the relevant data
    def load_data(self):
        X_train = pd.read_csv(f"{self.train_data_path}/X_train.csv")
        x_test = pd.read_csv(f"{self.test_data_path}/X_test.csv")
        Y_train = pd.read_csv(f"{self.train_data_path}/Y_train.csv").values.flatten()
        Y_test = pd.read_csv(f"{self.test_data_path}/Y_test.csv").values.flatten()

        return X_train, x_test, Y_train, Y_test

#Train the model
    def train_model(self, X_train, y_train):
        #Initial weights set to 0 including the intercept term
        initial_w = np.zeros(X_train.shape[1] + 1)

        #Optimize the cost function
        res = minimize(
            fun = lambda w: LogisticRegressionUtils.costFunction(w, X_train.values, y_train),
            x0 = initial_w,
            jac = True,
            method = 'TNC'
        )

        if res.success:
            print("Optimization completed succesfully.")
            self.weights = res.x
        else:
            raise ValueError("Optimization failed")
        
    def evaluate(self, X_test, y_test):
        #Add intercept term to test set
        X_test = np.concatenate([np.ones((X_test.shape[0], 1)), X_test.values], axis=1)

        #Make predictions
        probabilities = LogisticRegressionUtils.sigmoid(X_test @ self.weights)
        #Convert probabilities to binary predictions
        predictions = (probabilities >= 0.5).astype(int) 

        # Print evaluation metrics
        print("Classification Report:")
        print(classification_report(y_test, predictions))
        print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
        print(f"Precision: {precision_score(y_test, predictions, average='weighted'):.4f}")
        print(f"Recall: {recall_score(y_test, predictions, average='weighted'):.4f}")

    def save_model(self):
        #Save the weights to a file
        joblib.dump(self.weights, self.model_output_path)
        print(f"Model weights saved to {self.model_output_path}")

    def run_class_pipeline(self):
        print("Starting model training pipeline...")
        X_train, X_test, y_train, y_test = self.load_data()
        print("Data loaded successfully.")
        
        print("Training the model...")
        self.train(X_train, y_train)
        
        print("Evaluating the model...")
        self.evaluate(X_test, y_test)
        
        print("Saving the model...")
        self.save_model()
        print("Pipeline completed successfully.")

    

#Save the model ( and other data that might be of interest such as calculated weights)