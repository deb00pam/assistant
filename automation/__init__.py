"""
Automation Module - Storage and utilities

This module contains storage utilities and OS detection.
GUI automation components have been removed.
"""

# GUI automation components removed
# from .gui import ScreenAnalyzer, GUIAutomator

# Storage is still available
from .storage import load_user_data, save_user_data

# OS detection still available
try:
    from .os_detection import get_os_context, get_os_commands, os_detector
    OS_DETECTION_AVAILABLE = True
except ImportError:
    OS_DETECTION_AVAILABLE = False

__all__ = ['load_user_data', 'save_user_data']
