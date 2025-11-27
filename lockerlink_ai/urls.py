"""
URL configuration for LockerLink AI project.
"""
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from api.views import home_view

urlpatterns = [
    path('', home_view, name='home'),
    path('api/', include('api.urls')),
]

# Serve media files in development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATICFILES_DIRS[0] if settings.STATICFILES_DIRS else None)

