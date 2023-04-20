from django.contrib import admin
from django.urls import path, include

# a list of all the projects urls
urlpatterns = [
    # the url to the admin site
    path('admin/', admin.site.urls),
    # registering all the assistant application urls
    path('', include('assistant.urls')),
]
