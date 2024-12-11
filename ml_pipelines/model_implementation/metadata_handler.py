import os
import json
import requests
import subprocess
from datetime import datetime
from importlib.metadata import version
from dotenv import load_dotenv
load_dotenv()

URL = os.getenv("URL")

class DataMetadata:

    def __init__(self, strategy, model_pickle, scaler_pickle, description, database_class, training_date, metadata_file_path, performance_json_filepath):
        self.strategy = strategy
        self.model_pickle = model_pickle
        self.scaler_pickle = scaler_pickle
        self.description = description
        self.database_class = database_class
        self.training_date = training_date
        self.metadata_file_path = metadata_file_path
        self.performance_json_filepath = performance_json_filepath

    def prepare_metadata(self):
        metadata = {
            "strategy": self.strategy,
            "model_pickle": self.model_pickle,
            "scaler_pickle": self.scaler_pickle,
            "database_class": self.database_class,
            "description": self.description,
            "training date": self.training_date,
            }

        try:
            # Open and read the performance JSON file
            with open(self.performance_json_filepath, 'r') as file:
                performance_data = json.load(file)

            # Add performance data to metadata
            metadata["performance metrics"] = performance_data
        except FileNotFoundError:
            metadata["performance metrics"] = "Performance file not found."
        except json.JSONDecodeError:
            metadata["performance metrics"] = "Error decoding performance file."

        return metadata

    def save_metadata_to_file(self, metadata):
        with open(self.metadata_file_path, "w") as metadata_file:
            json.dump(metadata, metadata_file, indent=4)

        print(f"Successfully inserted data and saved metadata to {self.metadata_file_path}")

    def upload_metadata(self):
        # Prepare the metadata
        metadata = self.prepare_metadata()

        # Save metadata to file
        self.save_metadata_to_file(metadata)

        # Open the file in binary mode and upload it
        with open(self.metadata_file_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{URL}/upload_metadata/", files=files)

        # Print the response from the server
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")

        # Rename the model metadata file in django
        data = {
            'old_name': f"metadata_{self.strategy}.json",
            'new_name': f"data_{self.strategy.lower()}_{self.training_date}_metadata.json"
        }
        response = requests.post(f"{URL}/rename_metadata/", data=data)

        # remove the metadata file
        os.remove(self.metadata_file_path)

        # remove the performance file
        os.remove(self.performance_json_filepath)