# here we are import path from in-built django-urls
from django.urls import path
# here we are importing all the Views from the views.py file
from . import views

# a list of all the urls
urlpatterns = [
    path('home/', views.home, name='home'),
    path('error-handler/', views.error_handler, name='error_handler'),
]