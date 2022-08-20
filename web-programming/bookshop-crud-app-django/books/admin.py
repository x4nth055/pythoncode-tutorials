from django.contrib import admin
# from the models.py file import Book
from .models import Book

# registering the Book to the admin site
admin.site.register(Book)