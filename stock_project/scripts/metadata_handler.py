# Author: Nasit Vurgun
import os
import json
import requests
import subprocess
from datetime import datetime, timedelta
from importlib.metadata import version

class DataMetadata:
    def __init__(self, name, description, stocks, model, schema, start, end, url="http://127.0.0.1:8000/upload_metadata/"):
        self.name = name
        self.description = description
        self.stocks = stocks
        self.model = model
        self.schema = schema
        self.start = start
        self.end = end
        self.url = url
        self.metadata_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "metadata_StockData.json")

    def get_last_commit_hash(self):
        try:
            commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')
            return commit_hash
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")
            return None

    def prepare_metadata(self):
        metadata = {
            "name of file": self.name,
            "commit hash": self.get_last_commit_hash(),
            "model": self.model,
            "description": self.description,
            "schema": self.schema,
            "stocks": self.stocks,
            "origin": f"Alpaca API, library version: {version('alpaca-py') if version('alpaca-py') else 'unknown'}",
            "start date": str(self.start.date()),
            "end date": str(self.end.date()),
            "date collected": str(datetime.now())
        }
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
            response = requests.post(self.url, files=files)

        # Print the response from the server
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")

# Example usage:
# stocks = [list of stock data here]
# metadata_handler = DataMetadata(stocks)
# metadata_handler.upload_metadata()
