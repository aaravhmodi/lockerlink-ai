"""
API URL configuration for LockerLink AI.
"""
from django.urls import path
from . import views

urlpatterns = [
    path('', views.root, name='root'),
    path('test/', views.test_endpoint, name='test'),
    path('health/', views.health_check, name='health'),
    path('analyze/video/', views.analyze_video, name='analyze_video'),
    path('analyze/batch/', views.analyze_batch_videos, name='analyze_batch'),
    path('upload/', views.upload_video, name='upload_video'),
    path('test/swing/', views.test_swing_video, name='test_swing'),
]

