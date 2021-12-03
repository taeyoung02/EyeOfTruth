from django.urls import path, include
from . import views


urlpatterns = [
    path('analysis/', views.analysis),
    path('about_me/', views.about_me),
    path('', views.index),
]
