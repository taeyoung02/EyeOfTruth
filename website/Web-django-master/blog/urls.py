from django.urls import path, include
from . import views


urlpatterns = [
    path('', views.PostList.as_view()),
    path('upload/', views.upload_file)
]