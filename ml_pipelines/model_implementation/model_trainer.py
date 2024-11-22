import numpy as np
import pandas as pd


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
        Y_train = pd.read_csv(f"{self.train_data_path}/Y_train.csv")
        Y_test = pd.read_csv(f"{self.test_data_path}/Y_test.csv")

        return X_train, x_test, Y_train, Y_test

#Train the model
    def train_model(self, X_train, y_train):
        #Initial weights set to 0
        initial_w = np.zeros(X_train.shape[1] + 1)

        #Optimize the cost function
        res = minimize(
            fun = lambda w: LogisticRegressionUtils.costFunction(w, X_train.values, y_train),
            x0 = initial_w,
            
        )

#Save the model ( and other data that might be of interest such as calculated weights)