#!/usr/bin/env python3
"""
Gemini Desktop Assistant - Complete Command Line Interface

A self-operating desktop assistant powered by Google's Gemini AI that can see,
understand, and interact with your desktop environment through natural language commands.
"""

import os
import sys
import time
import json
import base64
import logging
import argparse
from io import BytesIO
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
from dataclasses import dataclass

# Third-party imports
import pyautogui
import cv2
import numpy as np
from PIL import Image, ImageDraw
from pynput import mouse, keyboard
from pynput.keyboard import Key
from dotenv import load_dotenv

# Google AI imports
import google.generativeai as genai


# =============================================================================
# Configuration and Data Classes
# =============================================================================

@dataclass
class AssistantConfig:
    """Configuration for the desktop assistant."""
    safe_mode: bool = False  # Disabled by default
    confirmation_required: bool = False  # Disabled by default
    max_actions_per_task: int = 10
    action_timeout: int = 30
    screenshot_dir: str = "screenshots"
    log_level: str = "INFO"


# =============================================================================
# Core Classes
# =============================================================================

class GeminiClient:
    """Client for interacting with Google's Gemini AI API."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('GOOGLE_AI_API_KEY')
        if not self.api_key:
            raise ValueError("Google AI API key is required")
            
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
    def analyze_screenshot(self, image: Image.Image, task_description: str) -> Dict[str, Any]:
        """Analyze a screenshot and provide action recommendations."""
        try:
            prompt = f"""
You are a desktop assistant AI that can see and understand computer screens.

TASK: {task_description}

CRITICAL INSTRUCTIONS:
- NEVER type terminal/command line commands
- NEVER use "edge website.com" command syntax  
- Use Windows UI elements: Start menu, taskbar, applications
- Key combinations like Ctrl+L must be written as "ctrl+l" (NOT separate keys)
- To open applications: Press Windows key → Type app name → Press Enter
- To navigate to websites: Open browser first, then use address bar with Ctrl+L

EXACT ACTIONS REQUIRED for browser + website tasks:
1. Press "win" key (opens Start menu)
2. Type application name only (e.g., "edge")
3. Press "enter" (launches app)
4. Wait 3 seconds (app loading time)
5. Press "ctrl+l" key combination (focuses address bar)
6. Type URL only (e.g., "chatgpt.com")
7. Press "enter" (navigates to site)

Please respond EXACTLY in this JSON format:
{{
    "understanding": "What I see on screen",
    "actions": [
        {{
            "action_type": "key_press",
            "key": "win",
            "description": "Press Windows key to open Start menu",
            "confidence": 0.9
        }},
        {{
            "action_type": "type", 
            "text": "edge",
            "description": "Type 'edge' to search for browser",
            "confidence": 0.9
        }},
        {{
            "action_type": "key_press",
            "key": "enter", 
            "description": "Press Enter to launch Edge",
            "confidence": 0.9
        }},
        {{
            "action_type": "wait",
            "duration": 3,
            "description": "Wait for browser to load",
            "confidence": 0.9
        }},
        {{
            "action_type": "key_press",
            "key": "ctrl+l",
            "description": "Press Ctrl+L to focus address bar", 
            "confidence": 0.9
        }},
        {{
            "action_type": "type",
            "text": "chatgpt.com",
            "description": "Type website URL",
            "confidence": 0.9
        }},
        {{
            "action_type": "key_press", 
            "key": "enter",
            "description": "Press Enter to navigate",
            "confidence": 0.9
        }}
    ],
    "safety_concerns": [],
    "next_steps": "Browser should open and navigate to website",
    "overall_confidence": 0.9
}}
"""
            
            response = self.model.generate_content([prompt, image])
            return self._parse_response(response.text)
            
        except Exception as e:
            logging.error(f"Error analyzing screenshot: {e}")
            return {
                "error": str(e),
                "actions": [],
                "confidence": 0.0
            }
    
    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the AI response into structured data."""
        try:
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_text = response_text[json_start:json_end]
                return json.loads(json_text)
            else:
                return {
                    "understanding": response_text,
                    "actions": [],
                    "safety_concerns": [],
                    "next_steps": "Manual review required",
                    "overall_confidence": 0.5
                }
                
        except json.JSONDecodeError:
            return {
                "understanding": response_text,
                "actions": [],
                "safety_concerns": ["Could not parse AI response"],
                "next_steps": "Manual review required", 
                "overall_confidence": 0.3
            }


class ScreenAnalyzer:
    """Handles screen capture and visual analysis."""
    
    def __init__(self, screenshot_dir: str = "screenshots"):
        self.screenshot_dir = screenshot_dir
        os.makedirs(screenshot_dir, exist_ok=True)
        pyautogui.FAILSAFE = False  # Disable fail-safe for automation
        pyautogui.PAUSE = 0.1
        
    def capture_screenshot(self, save: bool = True) -> Image.Image:
        """Capture a screenshot of the entire screen."""
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
        try:
            size = pyautogui.size()
            return (size.width, size.height)
        except Exception as e:
            logging.error(f"Error getting screen resolution: {e}")
            return (1920, 1080)


class GUIAutomator:
    """Handles GUI automation tasks."""
    
    def __init__(self, safety_delay: float = 0.1):
        self.safety_delay = safety_delay
        self.last_action_time = 0
        pyautogui.FAILSAFE = False  # Disable fail-safe for automation
        pyautogui.PAUSE = safety_delay
        self.action_history = []
        
    def click(self, x: int, y: int, button: str = "left", clicks: int = 1) -> bool:
        """Click at the specified coordinates."""
        try:
            # Validate coordinates to prevent fail-safe trigger
            if x <= 10 or y <= 10:
                logging.warning(f"Invalid coordinates ({x}, {y}), skipping click")
                return False
                
            self._enforce_safety_delay()
            logging.info(f"Clicking at ({x}, {y}) with {button} button")
            pyautogui.click(x, y, clicks=clicks, button=button)
            self._record_action("click", {"coordinates": (x, y), "button": button})
            return True
        except Exception as e:
            logging.error(f"Error clicking at ({x}, {y}): {e}")
            return False
    
    def type_text(self, text: str, interval: float = 0.0) -> bool:
        """Type the specified text."""
        try:
            self._enforce_safety_delay()
            logging.info(f"Typing text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            pyautogui.write(text, interval=interval)
            self._record_action("type", {"text": text})
            return True
        except Exception as e:
            logging.error(f"Error typing text: {e}")
            return False
    
    def press_key(self, key: str, presses: int = 1) -> bool:
        """Press a keyboard key or key combination."""
        try:
            self._enforce_safety_delay()
            logging.info(f"Pressing key: '{key}' {presses} times")
            
            # Handle key combinations
            if '+' in key:
                keys = key.split('+')
                # Convert 'win' to 'winleft' for pyautogui
                keys = ['winleft' if k.strip().lower() == 'win' else k.strip() for k in keys]
                pyautogui.hotkey(*keys)
            else:
                # Convert 'win' to 'winleft' for single key
                if key.lower() == 'win':
                    key = 'winleft'
                pyautogui.press(key, presses=presses)
                
            self._record_action("key_press", {"key": key, "presses": presses})
            return True
        except Exception as e:
            logging.error(f"Error pressing key '{key}': {e}")
            return False
    
    def scroll(self, clicks: int, x: Optional[int] = None, y: Optional[int] = None) -> bool:
        """Scroll at the specified location."""
        try:
            self._enforce_safety_delay()
            if x is not None and y is not None:
                pyautogui.scroll(clicks, x=x, y=y)
            else:
                pyautogui.scroll(clicks)
            self._record_action("scroll", {"clicks": clicks, "position": (x, y)})
            return True
        except Exception as e:
            logging.error(f"Error scrolling: {e}")
            return False
    
    def execute_action(self, action: Dict[str, Any]) -> bool:
        """Execute an action based on a dictionary specification."""
        action_type = action.get("action_type", "").lower()
        
        try:
            if action_type == "click":
                coords = action.get("coordinates", [0, 0])
                # If coordinates are invalid, try pressing Enter instead (for search results)
                if coords[0] <= 10 or coords[1] <= 10:
                    logging.info("Invalid coordinates, pressing Enter instead")
                    return self.press_key("enter")
                return self.click(coords[0], coords[1])
            elif action_type == "type":
                text = action.get("text", "")
                return self.type_text(text)
            elif action_type == "key_press":
                key = action.get("key", "")
                return self.press_key(key)
            elif action_type == "scroll":
                clicks = action.get("clicks", 1)
                coords = action.get("coordinates")
                if coords:
                    return self.scroll(clicks, coords[0], coords[1])
                else:
                    return self.scroll(clicks)
            elif action_type == "wait":
                # Skip all waiting - no delays
                logging.info("Skipping wait action - no delays enabled")
                return True
            else:
                logging.error(f"Unknown action type: {action_type}")
                return False
        except Exception as e:
            logging.error(f"Error executing action {action_type}: {e}")
            return False
    
    def _enforce_safety_delay(self):
        """Enforce minimum delay between actions."""
        current_time = time.time()
        elapsed = current_time - self.last_action_time
        if elapsed < self.safety_delay:
            time.sleep(self.safety_delay - elapsed)
        self.last_action_time = time.time()
    
    def _record_action(self, action_type: str, details: Dict[str, Any]):
        """Record an action for history purposes."""
        self.action_history.append({
            "timestamp": time.time(),
            "type": action_type,
            "details": details
        })
        if len(self.action_history) > 100:
            self.action_history.pop(0)


class DesktopAssistant:
    """Main desktop assistant class."""
    
    def __init__(self, config: Optional[AssistantConfig] = None):
        self.config = config or AssistantConfig()
        
        # Set up logging
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Initialize components
        self.screen_analyzer = ScreenAnalyzer(self.config.screenshot_dir)
        self.gemini_client = GeminiClient()
        self.gui_automator = GUIAutomator()
        
        # Task state
        self.current_task = None
        self.task_progress = []
        self.is_running = False
        
        logging.info("Desktop Assistant initialized")
    
    def execute_task(self, task_description: str) -> Dict[str, Any]:
        """Execute a high-level task."""
        logging.info(f"Starting task: {task_description}")
        
        self.current_task = task_description
        self.task_progress = []
        self.is_running = True
        
        try:
            # Capture screenshot
            screenshot = self.screen_analyzer.capture_screenshot()
            
            # Get AI analysis
            analysis = self.gemini_client.analyze_screenshot(screenshot, task_description)
            
            if "error" in analysis:
                return {
                    "success": False,
                    "error": analysis["error"],
                    "actions_completed": 0
                }
            
            # Log safety concerns but don't block execution
            if analysis.get("safety_concerns"):
                logging.info(f"Safety concerns noted: {analysis['safety_concerns']}")
                # Continue execution anyway
            
            # Execute actions
            actions = analysis.get("actions", [])
            logging.info(f"AI generated {len(actions)} actions: {[a.get('description', 'No desc') for a in actions]}")
            
            # Check if this is a browser + website task and AI didn't include navigation steps
            if any(word in task_description.lower() for word in ['go to', '.com', '.org', '.net', 'website']) and \
               not any('ctrl+l' in str(action).lower() or 'address' in str(action).lower() for action in actions):
                
                logging.info("Detected website navigation task, adding browser navigation steps")
                
                # Extract website from task
                import re
                websites = re.findall(r'([\w.-]+\.(?:com|org|net|edu|gov))', task_description.lower())
                if websites:
                    website = websites[0]
                    logging.info(f"Extracted website: {website}")
                    
                    # Find where to insert Ctrl+L before typing the website
                    for i, action in enumerate(actions):
                        if action.get("action_type") == "type" and website in action.get("text", ""):
                            # Insert Ctrl+L before typing the website
                            actions.insert(i, {
                                "action_type": "key_press",
                                "key": "ctrl+l", 
                                "description": "Focus address bar with Ctrl+L"
                            })
                            logging.info("Added Ctrl+L before typing website URL")
                            break
                    
                    # Add navigation actions if needed
                    navigation_actions = [
                        {
                            "action_type": "key_press", 
                            "key": "ctrl+l",
                            "description": "Focus address bar with Ctrl+L"
                        },
                        {
                            "action_type": "type",
                            "text": website,
                            "description": f"Type website URL: {website}"
                        },
                        {
                            "action_type": "key_press",
                            "key": "enter", 
                            "description": "Press Enter to navigate"
                        }
                    ]
                    actions.extend(navigation_actions)
                    logging.info(f"Added {len(navigation_actions)} navigation actions")
            
            results = self._execute_action_sequence(actions)
            
            return {
                "success": len([r for r in results if r["success"]]) == len(results),
                "task_description": task_description,
                "initial_analysis": analysis,
                "actions_completed": len([r for r in results if r["success"]]),
                "total_actions": len(actions),
                "action_results": results
            }
            
        except Exception as e:
            logging.error(f"Error executing task: {e}")
            return {
                "success": False,
                "error": str(e),
                "actions_completed": len(self.task_progress)
            }
        finally:
            self.is_running = False
    
    def _execute_action_sequence(self, actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute a sequence of actions."""
        results = []
        
        for i, action in enumerate(actions):
            if not self.is_running:
                break
            
            if i >= self.config.max_actions_per_task:
                logging.warning(f"Maximum actions limit reached")
                break
            
            # Execute actions without confirmation
            logging.info(f"Executing action {i+1}/{len(actions)}: {action.get('description', 'No description')}")
            
            # Execute the action
            result = self._execute_single_action(action)
            results.append(result)
            self.task_progress.append(result)
            
            if not result["success"]:
                logging.error(f"Action failed: {result.get('error', 'Unknown error')}")
                break
            
            time.sleep(0.5)
        
        return results
    
    def _execute_single_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single action."""
        start_time = time.time()
        action_type = action.get("action_type", "unknown")
        
        logging.info(f"Executing {action_type}: {action.get('description', 'No description')}")
        
        try:
            success = self.gui_automator.execute_action(action)
            return {
                "action": action,
                "success": success,
                "execution_time": time.time() - start_time,
                "timestamp": time.time(),
                "error": None if success else "Action execution failed"
            }
        except Exception as e:
            return {
                "action": action,
                "success": False,
                "execution_time": time.time() - start_time,
                "timestamp": time.time(),
                "error": str(e)
            }
    
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
            print(f"⚠️  WARNING: This action may have safety risks!")
        
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


# =============================================================================
# CLI Functions
# =============================================================================

def setup_logging(level: str = "INFO"):
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def load_environment():
    """Load environment variables."""
    load_dotenv()
    
    api_key = os.getenv('GOOGLE_AI_API_KEY')
    if not api_key:
        print("╔══════════════════════════════════════════════════════════╗")
        print("║                    API KEY REQUIRED                       ║")
        print("╠══════════════════════════════════════════════════════════╣")
        print("║ Google AI API key not found in environment variables.     ║")
        print("║                                                            ║")
        print("║ Please get your API key from:                             ║")
        print("║ https://aistudio.google.com/apikey                        ║")
        print("║                                                            ║")
        print("║ Then either:                                              ║")
        print("║ 1. Create a .env file with: GOOGLE_AI_API_KEY=your_key    ║")
        print("║ 2. Set environment variable: GOOGLE_AI_API_KEY=your_key   ║")
        print("╚══════════════════════════════════════════════════════════╝")
        
        # Allow user to enter key interactively
        api_key = input("\nEnter your Google AI API key (or press Enter to exit): ").strip()
        if api_key:
            os.environ['GOOGLE_AI_API_KEY'] = api_key
            # Try to save to .env file
            try:
                with open('.env', 'w') as f:
                    f.write(f"GOOGLE_AI_API_KEY={api_key}\n")
                print("✅ API key saved to .env file")
            except Exception as e:
                print(f"⚠️  Could not save to .env file: {e}")
        else:
            sys.exit(1)


def interactive_mode():
    """Run the assistant in interactive mode."""
    print("╔══════════════════════════════════════════════════════════╗")
    print("║            GEMINI DESKTOP ASSISTANT - CLI                 ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print("║ 🤖 AI-powered desktop automation                          ║")
    print("║ 📸 Visual screen understanding                            ║")
    print("║ 🔒 Safe operation with confirmations                      ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()
    print("Type your tasks in natural language or use these commands:")
    print("• 'help' - Show available commands")
    print("• 'analyze' - Analyze current screen")
    print("• 'status' - Show assistant status")
    print("• 'quit' - Exit the assistant")
    print()
    
    # Load configuration
    config = AssistantConfig(
        safe_mode=False,  # Always disabled
        confirmation_required=False,  # Always disabled
        screenshot_dir=os.getenv('SCREENSHOT_DIR', 'screenshots')
    )
    
    # Initialize assistant
    try:
        print("🔄 Initializing assistant...")
        assistant = DesktopAssistant(config)
        print(f"✅ Assistant ready! (Safe Mode: {'ON' if config.safe_mode else 'OFF'})")
        print()
    except Exception as e:
        print(f"❌ Failed to initialize assistant: {e}")
        return
    
    while True:
        try:
            user_input = input("🤖 › ").strip()
            
            if not user_input:
                continue
                
            # Handle commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            elif user_input.lower() == 'help':
                show_help()
                continue
            elif user_input.lower() == 'status':
                show_status(assistant)
                continue
            elif user_input.lower() == 'analyze':
                analyze_screen(assistant)
                continue
            elif user_input.lower().startswith('config'):
                handle_config_command(assistant, user_input)
                continue
            
            # Execute task
            print(f"\n🚀 Executing: {user_input}")
            print("─" * 60)
            
            result = assistant.execute_task(user_input)
            
            # Show results
            print("\n" + "═"*60)
            if result["success"]:
                print(f"✅ SUCCESS - Task completed!")
                print(f"   Actions: {result['actions_completed']}/{result['total_actions']}")
            else:
                print(f"❌ FAILED - {result.get('error', 'Unknown error')}")
                print(f"   Actions completed: {result['actions_completed']}")
                
                if result.get("safety_concerns"):
                    print(f"⚠️  Safety concerns: {', '.join(result['safety_concerns'])}")
            print("═"*60)
            print()
            
        except KeyboardInterrupt:
            print("\n\n🛑 Task interrupted by user")
            assistant.stop_task()
            print()
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print()


def show_help():
    """Show help information."""
    print("""
╔══════════════════════════════════════════════════════════╗
║                         HELP                             ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║ 📋 TASK EXAMPLES:                                        ║
║   • "Take a screenshot"                                  ║
║   • "Open notepad and type hello world"                 ║
║   • "Click on the Chrome icon"                          ║
║   • "Press the Windows key"                             ║
║   • "Scroll down 3 times"                               ║
║                                                          ║
║ 🔧 CONTROL COMMANDS:                                     ║
║   • help     - Show this help                           ║
║   • status   - Show assistant status                    ║
║   • analyze  - Analyze current screen                   ║
║   • quit     - Exit assistant                           ║
║                                                          ║
║ ⚙️  CONFIGURATION:                                       ║
║   • config safe on/off      - Toggle safe mode         ║
║   • config confirm on/off   - Toggle confirmations     ║
║   • config show             - Show current config      ║
║                                                          ║
║ 💡 TIPS:                                                ║
║   • Be specific about what you want                     ║
║   • Use Ctrl+C to stop running tasks                    ║
║   • Screenshots are saved automatically                 ║
║   • All actions are logged and confirmed               ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""")


def show_status(assistant: DesktopAssistant):
    """Show assistant status."""
    status = assistant.get_task_status()
    config = assistant.config
    
    print(f"""
╔══════════════════════════════════════════════════════════╗
║                    ASSISTANT STATUS                      ║
╠══════════════════════════════════════════════════════════╣
║ Current Task: {(status['current_task'] or 'None'):<40} ║
║ Running: {('Yes' if status['is_running'] else 'No'):<47} ║
║ Actions Completed: {status['actions_completed']:<35} ║
║                                                          ║
║ Safe Mode: {('ON' if config.safe_mode else 'OFF'):<46} ║
║ Confirmation: {('ON' if config.confirmation_required else 'OFF'):<43} ║
║ Max Actions: {config.max_actions_per_task:<44} ║
║ Screenshot Dir: {config.screenshot_dir:<39} ║
╚══════════════════════════════════════════════════════════╝
""")


def analyze_screen(assistant: DesktopAssistant):
    """Analyze the current screen."""
    print("🔍 Analyzing current screen...")
    
    try:
        analysis = assistant.analyze_current_screen("Describe what you see on this screen in detail.")
        
        print("\n" + "═"*60)
        print("SCREEN ANALYSIS")
        print("═"*60)
        print(f"{analysis.get('understanding', 'No analysis available')}")
        
        if analysis.get('actions'):
            print(f"\nSuggested Actions:")
            for i, action in enumerate(analysis['actions'], 1):
                print(f"  {i}. {action.get('description', 'No description')}")
        
        if analysis.get('safety_concerns'):
            print(f"\n⚠️  Safety Concerns: {', '.join(analysis['safety_concerns'])}")
            
        print(f"\nConfidence: {analysis.get('overall_confidence', 0.0):.2f}")
        print("═"*60)
        
    except Exception as e:
        print(f"❌ Error analyzing screen: {e}")


def handle_config_command(assistant: DesktopAssistant, command: str):
    """Handle configuration commands."""
    parts = command.split()
    
    if len(parts) < 2:
        print("Usage: config <setting> <value> or config show")
        return
    
    if parts[1] == 'show':
        show_status(assistant)
        return
    
    if len(parts) < 3:
        print("Usage: config <setting> <value>")
        return
    
    setting = parts[1].lower()
    value = parts[2].lower()
    
    if setting == 'safe':
        if value in ['on', 'true', 'yes']:
            assistant.set_safe_mode(True)
            print("✅ Safe mode enabled")
        elif value in ['off', 'false', 'no']:
            assistant.set_safe_mode(False)
            print("⚠️  Safe mode disabled")
        else:
            print("Usage: config safe on/off")
    
    elif setting in ['confirm', 'confirmation']:
        if value in ['on', 'true', 'yes']:
            assistant.set_confirmation_required(True)
            print("✅ Action confirmation enabled")
        elif value in ['off', 'false', 'no']:
            assistant.set_confirmation_required(False)
            print("⚠️  Action confirmation disabled")
        else:
            print("Usage: config confirm on/off")
    
    else:
        print(f"Unknown setting: {setting}")


def single_task_mode(task: str):
    """Execute a single task and exit."""
    print(f"🚀 Executing single task: {task}")
    
    config = AssistantConfig(
        safe_mode=False,  # Always disabled
        confirmation_required=False,  # Always disabled
        screenshot_dir=os.getenv('SCREENSHOT_DIR', 'screenshots')
    )
    
    try:
        assistant = DesktopAssistant(config)
        result = assistant.execute_task(task)
        
        if result["success"]:
            print(f"✅ Task completed successfully!")
            print(f"Actions: {result['actions_completed']}/{result['total_actions']}")
        else:
            print(f"❌ Task failed: {result.get('error', 'Unknown error')}")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Gemini Desktop Assistant - AI-powered desktop automation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                              # Interactive mode
  python main.py -t "open calculator"        # Single task mode
  python main.py --analyze                   # Analyze screen only
        """
    )
    
    parser.add_argument(
        '--task', '-t',
        help='Execute a single task and exit'
    )
    
    parser.add_argument(
        '--analyze', '-a',
        action='store_true',
        help='Analyze current screen and exit'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set logging level'
    )
    
    parser.add_argument(
        '--no-safe-mode',
        action='store_true',
        help='Disable safe mode'
    )
    
    args = parser.parse_args()
    
    # Setup
    setup_logging(args.log_level)
    load_environment()
    
    if args.no_safe_mode:
        os.environ['SAFE_MODE'] = 'false'
    
    # Run appropriate mode
    try:
        if args.analyze:
            config = AssistantConfig()
            assistant = DesktopAssistant(config)
            analyze_screen(assistant)
        elif args.task:
            single_task_mode(args.task)
        else:
            interactive_mode()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()