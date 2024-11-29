from django.urls import path
from stock_app import views, api_views
# Add home to path
urlpatterns = [
    path('', views.home, name='home'), # Render home html page
    path('db_updates/', api_views.upload_csv, name='upload_csv'), # endpoint for uploading macd data
    path('db_updates/ema/', api_views.upload_ema, name='upload_ema'),
    path('db_updates/rsi/', api_views.upload_rsi, name='upload_rsi'), 
    path('get_database/macd/', api_views.get_macd_data, name='get_macd'), 
    path('get_database/ema/', api_views.get_ema_data, name='get_ema'),
    path('get_database/rsi/', api_views.get_rsi_data, name='get_rsi'),
    
]
