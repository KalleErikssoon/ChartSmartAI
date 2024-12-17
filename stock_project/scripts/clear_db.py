# Author: Isaac Lindegren Ternbom
# Django imports
import os
import sys
import django

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'stock_project.settings')
django.setup()

from stock_app.models import StockData

class DatabaseCleaner:
    """
    A class responsible for clearing all rows from the StockData table in the database.
    """

    @staticmethod
    def clear_table():
        """
        Clears all rows from the StockData table.
        """
        try:
            # Delete all records from the StockData table
            StockData.objects.all().delete()
            print("All rows cleared from the StockData table successfully.")

        except Exception as e:
            print(f"An error occurred while clearing the table: {e}")

# Main block to execute the cleaner
if __name__ == "__main__":
    cleaner = DatabaseCleaner()
    cleaner.clear_table()