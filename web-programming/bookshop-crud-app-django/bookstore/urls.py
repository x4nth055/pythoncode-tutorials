# importing the django's in-built admin url
from django.contrib import admin
# importing path and include from django's in-built urls
from django.urls import path, include

# importing conf from settings.py
from django.conf import settings
# importing conf.urls from static
from django.conf.urls.static import static

# defining the list for urls
urlpatterns = [
    path('admin/', admin.site.urls),
    # registering books application's urls in project
    path('bookstore/', include('books.urls')),
]
# appending the urls with the static urls
urlpatterns += static(settings.MEDIA_URL, document_root = settings.MEDIA_ROOT)