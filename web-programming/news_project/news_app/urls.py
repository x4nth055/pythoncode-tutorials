from django.urls import path
from .views import JournalistView, ArticleView, ArticleDetailView

app_name="news_app"

urlpatterns=[
    path('journalist/', JournalistView.as_view() ),
    path('article/', ArticleView.as_view() ),
    path('article/<int:pk>/', ArticleDetailView.as_view()),
]
