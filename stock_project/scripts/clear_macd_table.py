# Author: Isaac Lindegren Ternbom, Karl Eriksson
# Django imports
import os
import sys
import django

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'stock_project.settings')
django.setup()

from stock_app.models import MACD_Data

class MacdCleaner:
    """
    A class responsible for clearing all rows from the StockData table in the database.
    """

    @staticmethod
    def clear_table():
        """
        Clears all rows from the StockData table.
        """
        try:
                # check if table is empty
                if MACD_Data.objects.exists():
                    # delete all records from the table
                    MACD_Data.objects.all().delete()
                    print("All rows deleted from the MACD table successfully.")
                else:
                    print("The EMA table is empty.")

        except Exception as e:
                print(f"An error occurred while clearing the table: {e}")

# Main block to execute the cleaner
if __name__ == "__main__":
    cleaner = MacdCleaner()
    cleaner.clear_table()