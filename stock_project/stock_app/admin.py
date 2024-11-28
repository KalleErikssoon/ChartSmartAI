from django.contrib import admin
from .models import StockData, RSI_Data

# The purpose of this class is to create an admin interface for the StockData model
# This will allow us to view and manage the StockData objects in the Django admin UI (CRUD operations)

# Register your models here.
admin.site.register(StockData)
admin.site.register(RSI_Data)
