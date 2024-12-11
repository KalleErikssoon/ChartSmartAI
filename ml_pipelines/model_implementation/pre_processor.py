import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import requests
from io import StringIO 
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

class Preprocessor:
    def __init__(self, api_url, test_size=0.2, random_state=None, apply_smote=True, apply_scaling=True):
        self.api_url = api_url
        self.test_size = test_size
        self.random_state = random_state
        self.apply_smote = apply_smote
        self.apply_scaling = apply_scaling
        self.scaler = StandardScaler() if apply_scaling else None

    def fetch_data(self):
        # Make a GET request to the API
        response = requests.get(self.api_url)

        if response.status_code == 200:
            # Load the CSV response into a dataframe
            csv_data = StringIO(response.text)
            try:
                data = pd.read_csv(csv_data)
                if data.empty:
                    raise ValueError("No data found in the CSV file.")
                return data
            except pd.errors.EmptyDataError:
                raise ValueError("The CSV file is empty.")
            except Exception as e:
                raise ValueError(f"Failed to parse CSV data: {e}")
        else:
            # Handle errors
            raise Exception(f"Failed to fetch data: {response.status_code} - {response.text}")

    
    # def split_and_preprocess_data(self, X, y, stratify=None):
    #     """
    #     Split the data into training and test sets, apply SMOTE to balance the dataset, and scale features.
    #     """
    #     # Step 1: Split data into training and testing sets
    #     X_train, X_test, y_train, y_test = train_test_split(
    #         X, y, test_size=self.test_size, stratify=stratify, random_state=self.random_state
    #     )

    #     # Step 2: Apply SMOTE if enabled
    #     if self.apply_smote:
    #         print("Applying SMOTE to balance the dataset...")
    #         smote = SMOTE(random_state=self.random_state)
    #         X_train, y_train = smote.fit_resample(X_train, y_train)
    #     else:
    #         print("Skipping SMOTE...")

    #     # Step 3: Apply scaling if enabled
    #     if self.apply_scaling:
    #         print("Scaling features...")
    #         X_train = self.scaler.fit_transform(X_train)
    #         X_test = self.scaler.transform(X_test)
    #         # Convert scaled data back to DataFrame for consistency
    #         X_train = pd.DataFrame(X_train, columns=X.columns)
    #         X_test = pd.DataFrame(X_test, columns=X.columns)
    #     else:
    #         print("Skipping feature scaling...")

    #     return X_train, X_test, y_train, y_test

    def split_and_preprocess_data(self, X, y, stratify=None):
        """
        Split the data into training and test sets, apply SMOTE to balance the dataset, and scale features.
        """
        #apply scaling before the splitting
        if self.apply_scaling:
            X_scaled = self.scaler.fit_transform(X)
            scaler_mean = self.scaler.mean_
            scaler_scale = self.scaler.scale_
            print(f"Scaler mean:{scaler_mean}")
            print(f"Scaler scale:{scaler_scale}")


        else:
            print("Skipping feature scaling...")

        # Step 1: Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=self.test_size, stratify=stratify, random_state=self.random_state
        )

        # Step 2: Apply SMOTE if enabled
        if self.apply_smote:
            print("Applying SMOTE to balance the dataset...")
            smote = SMOTE(random_state=self.random_state)
            X_train, y_train = smote.fit_resample(X_train, y_train)
        else:
            print("Skipping SMOTE...")

        # # Step 3: Apply scaling if enabled
        # if self.apply_scaling:
        #     print("Scaling features...")
        #     X_train = self.scaler.fit_transform(X_train)
        #     X_test = self.scaler.transform(X_test)
        #     # Convert scaled data back to DataFrame for consistency
        X_train = pd.DataFrame(X_train, columns=X.columns)
        X_test = pd.DataFrame(X_test, columns=X.columns)
        # else:
        #     print("Skipping feature scaling...")

        return X_train, X_test, y_train, y_test