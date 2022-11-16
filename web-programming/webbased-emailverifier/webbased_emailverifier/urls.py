
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    # this points to admin.site urls
    path('admin/', admin.site.urls),
    # this points to verifier urls
    path('', include('verifier.urls')),
]
