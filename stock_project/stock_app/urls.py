from django.urls import path
from stock_app import views, api_views
# Add home to path
urlpatterns = [
    path('', views.home, name='home'), # Render home html page
    path('db_updates/', api_views.upload_csv, name='upload_csv') # endpoint for uploading macd data
]

urlpatterns = [
    path('', views.home, name='home'), # Render home html page
    path('db_updates/ema/', api_views.upload_ema, name='upload_ema')
]
