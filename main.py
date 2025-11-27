#!/usr/bin/env python
"""
Simple entry point to run Django development server.
Usage: python main.py runserver
"""

import os
import sys
import django
from django.core.management import execute_from_command_line

if __name__ == '__main__':
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'lockerlink_ai.settings')
    django.setup()
    
    # Default to runserver if no arguments provided
    if len(sys.argv) == 1:
        sys.argv.append('runserver')
    
    execute_from_command_line(sys.argv)

