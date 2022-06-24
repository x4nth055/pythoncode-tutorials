"""djangodictionary URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
# importing the django's in-built admin url
from django.contrib import admin
# importing path and include from django's in-built urls
from django.urls import path, include

# defining the list for urls
urlpatterns = [
    path('admin/', admin.site.urls),
    # registering dictionary app urls in project
    path('', include('dictionary.urls')),
]
