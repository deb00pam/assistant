"""
Desktop Assistant - Main assistant class for Truvo

This module contains the DesktopAssistant class that orchestrates all components
to execute desktop automation tasks and conversational interactions.
"""

import os
import sys
import time
import logging
from typing import Dict, List, Any, Optional

# Import required components with fallbacks
try:
    from llm.gemini.client import GeminiClient
except ImportError:
    GeminiClient = None

try:
    from interfaces.voice import VoiceHandler
except ImportError:
    VoiceHandler = None

try:
    from automation.gui import ScreenAnalyzer
except ImportError:
    ScreenAnalyzer = None

try:
    from automation.gui import GUIAutomator
except ImportError:
    GUIAutomator = None

# Try to import pyautogui for screen size detection
try:
    import pyautogui
except ImportError:
    pyautogui = None

# Import AssistantConfig from core.config
try:
    from core.config import AssistantConfig
except ImportError:
    AssistantConfig = None


class DesktopAssistant:
    """Main desktop assistant class."""
    
    def __init__(self, config: Optional['AssistantConfig'] = None):
        """Initialize the DesktopAssistant.
        
        Args:
            config: AssistantConfig instance. If None, creates a default config.
        """
        # Use provided config or create default
        if config:
            self.config = config
        elif AssistantConfig:
            self.config = AssistantConfig()
        else:
            # Fallback minimal config if AssistantConfig import failed
            from dataclasses import dataclass
            @dataclass
            class MinimalConfig:
                screenshot_dir: str = "screenshots"
                voice_enabled: bool = False
                log_level: str = "ERROR"
                safe_mode: bool = False
                confirmation_required: bool = False
                max_actions_per_task: int = 10
            self.config = MinimalConfig()
        
        # Set up logging
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
            
        if GeminiClient:
            self.gemini_client = GeminiClient()
        else:
            self.gemini_client = None
            
        if GUIAutomator:
            self.gui_automator = GUIAutomator()
        else:
            self.gui_automator = None
            
        self.voice_handler = VoiceHandler(self.config) if (VoiceHandler and self.config.voice_enabled) else None
        
        # Task state
        self.current_task = None
        self.task_progress = []
        self.is_running = False
        
        logging.info("Truvo Desktop Assistant initialized")
    
  
    def _confirm_action(self, action: Dict[str, Any]) -> bool:
        """Ask user to confirm an action."""
        print(f"\n{'='*60}")
        print("ACTION CONFIRMATION REQUIRED")
        print(f"{'='*60}")
        print(f"Action Type: {action.get('action_type', 'Unknown')}")
        print(f"Description: {action.get('description', 'No description')}")
        
        if action.get("coordinates"):
            print(f"Coordinates: {action['coordinates']}")
        if action.get("text"):
            print(f"Text to type: '{action['text']}'")
        if action.get("key"):
            print(f"Key to press: {action['key']}")
        
        safety_risk = action.get("safety_risk", False)
        if safety_risk:
            print(f"WARNING: This action may have safety risks!")
        
        confidence = action.get("confidence", 0.0)
        print(f"AI Confidence: {confidence:.2f}")
        
        while True:
            response = input("\nExecute this action? [y/n/s/v]: ").lower().strip()
            
            if response in ['y', 'yes']:
                return True
            elif response in ['n', 'no']:
                return False
            elif response in ['s', 'stop']:
                self.stop_task()
                return False
            elif response in ['v', 'view']:
                screenshot = self.screen_analyzer.capture_screenshot(save=True)
                print(f"Screenshot saved to: {self.config.screenshot_dir}")
                continue
            else:
                print("Invalid choice. Please enter y, n, s, or v.")
    
    def stop_task(self):
        """Stop the currently running task."""
        logging.info("Stopping current task...")
        self.is_running = False
    
    def get_task_status(self) -> Dict[str, Any]:
        """Get current task status."""
        return {
            "current_task": self.current_task,
            "is_running": self.is_running,
            "actions_completed": len(self.task_progress),
            "last_action": self.task_progress[-1] if self.task_progress else None
        }
    
    def set_safe_mode(self, enabled: bool):
        """Enable or disable safe mode."""
        self.config.safe_mode = enabled
        logging.info(f"Safe mode {'enabled' if enabled else 'disabled'}")
    
    def set_confirmation_required(self, required: bool):
        """Enable or disable action confirmation."""
        self.config.confirmation_required = required
        logging.info(f"Action confirmation {'enabled' if required else 'disabled'}")
    
    def analyze_current_screen(self, task_context: str = "") -> Dict[str, Any]:
        """Analyze the current screen without performing actions."""
        screenshot = self.screen_analyzer.capture_screenshot()
        prompt = task_context or "Please analyze this screenshot and describe what you see."
        analysis = self.gemini_client.analyze_screenshot(screenshot, prompt)
        return analysis
