"""
WSGI config for LockerLink AI project.
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'lockerlink_ai.settings')

application = get_wsgi_application()

