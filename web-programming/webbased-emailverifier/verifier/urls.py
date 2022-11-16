# from the current folder import views
from . import views
# importing path from django.urls
from django.urls import path

# this is the list of the app's views
# if the app has several views then it will have several paths
urlpatterns = [
    path('', views.index, name='home'),
]