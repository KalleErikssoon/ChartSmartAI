from django.urls import path
from stock_app import views
# Add home to path
urlpatterns = [
    path('', views.home, name='home'),
]
