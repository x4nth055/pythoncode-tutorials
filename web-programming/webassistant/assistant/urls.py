# here we are import path from in-built django-urls
from django.urls import path
# here we are importing all the Views from the views.py file
from . import views

# a list of all the urls
urlpatterns = [
    path('', views.home, name='home'),
    path('new_chat/', views.new_chat, name='new_chat'),
    path('error-handler/', views.error_handler, name='error_handler'),
]