# here we are import path from in-built django-urls
from django.urls import path

# here we are importing all the Views from the views.py file
from . import views

urlpatterns = [
    path('', views.index, name='home'),
]