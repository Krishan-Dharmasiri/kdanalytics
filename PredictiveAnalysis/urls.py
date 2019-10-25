from django.urls import path
from . import views

urlpatterns = [
    path("lynear/",views.lynear_regression,name='lynear_regression'),
    path('',views.lynear_regression,name='lynear_regression'),
    
]