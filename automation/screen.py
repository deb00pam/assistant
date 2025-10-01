"""
Screen Analyzer - Screen capture and visual analysis for Truvo

This module handles capturing screenshots and getting screen information.
"""

import os
import logging
from typing import Tuple
from datetime import datetime

# Import dependencies with fallbacks
try:
    import pyautogui
except ImportError:
    print("Warning: pyautogui not installed. Install with: pip install pyautogui")
    pyautogui = None

try:
    from PIL import Image
except ImportError:
    print("Warning: PIL not installed. Install with: pip install Pillow")
    Image = None


class ScreenAnalyzer:
    """Handles screen capture and visual analysis."""
    
    def __init__(self, screenshot_dir: str = "screenshots"):
        self.screenshot_dir = screenshot_dir
        os.makedirs(screenshot_dir, exist_ok=True)
        if pyautogui:
            pyautogui.FAILSAFE = False  # Disable fail-safe for automation
            pyautogui.PAUSE = 0.1
        
    def capture_screenshot(self, save: bool = True):
        """Capture a screenshot of the entire screen."""
        if not pyautogui:
            raise RuntimeError("pyautogui is not installed")
            
        try:
            screenshot = pyautogui.screenshot()
            
            if save:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = os.path.join(self.screenshot_dir, f"screenshot_{timestamp}.png")
                screenshot.save(filepath)
                logging.info(f"Screenshot saved to {filepath}")
                
            return screenshot
            
        except Exception as e:
            logging.error(f"Error capturing screenshot: {e}")
            raise
    
    def get_screen_resolution(self) -> Tuple[int, int]:
        """Get the current screen resolution."""
        if not pyautogui:
            return (1920, 1080)  # Default fallback
            
        try:
            size = pyautogui.size()
            return (size.width, size.height)
        except Exception as e:
            logging.error(f"Error getting screen resolution: {e}")
            return (1920, 1080)
