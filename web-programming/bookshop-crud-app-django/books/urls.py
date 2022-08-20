from django.urls import path
# this imports all the views from the views.py
from . import views


urlpatterns = [
    # this is the home url
    path('', views.home, name='home'),
    # this is the single book url
    path('book-detail/<str:id>/', views.book_detail, name='book-detail'),
    # this is the add book url
    path('add-book/', views.add_book, name='add-book'),
    # this is the edit book url
    path('edit-book/<str:id>/', views.edit_book, name='edit-book'),
    # this is the delete book url
    path('delete-book/<str:id>/', views.delete_book, name='delete-book'),
]