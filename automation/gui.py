"""
GUI Automator and Screen Analyzer - GUI automation and screen capture for Truvo

This module handles automated GUI interactions like clicking, typing, scrolling,
and screen capture functionality.
"""

import os
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# Import dependencies with fallbacks
try:
    import pyautogui
except ImportError:
    print("Warning: pyautogui not installed. Install with: pip install pyautogui")
    pyautogui = None


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
            
            filepath = None
            if save:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = os.path.join(self.screenshot_dir, f"screenshot_{timestamp}.png")
                screenshot.save(filepath)
                logging.info(f"Screenshot saved to {filepath}")
                
            return screenshot, filepath
            
        except Exception as e:
            logging.error(f"Error capturing screenshot: {e}")
            raise


class GUIAutomator:
    """Handles GUI automation tasks."""
    
    def __init__(self, safety_delay: float = 0.1):
        self.safety_delay = safety_delay
        self.last_action_time = 0
        if pyautogui:
            pyautogui.FAILSAFE = False  # Disable fail-safe for automation
            pyautogui.PAUSE = safety_delay
        self.action_history = []
        
    def click(self, x: int, y: int, button: str = "left", clicks: int = 1) -> bool:
        """Click at the specified coordinates."""
        if not pyautogui:
            logging.error("pyautogui is not installed")
            return False
            
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
        if not pyautogui:
            logging.error("pyautogui is not installed")
            return False
            
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
        if not pyautogui:
            logging.error("pyautogui is not installed")
            return False
            
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
        if not pyautogui:
            logging.error("pyautogui is not installed")
            return False
            
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
    
    def execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an action based on a dictionary specification and verify it worked."""
        action_type = action.get("action_type", "").lower()
        
        try:
            if action_type == "click":
                coords = action.get("coordinates", [0, 0])
                logging.info(f"Click action received with coordinates: {coords}")

                # Handle different coordinate formats
                if isinstance(coords, list) and len(coords) >= 2:
                    x, y = coords[0], coords[1]
                elif isinstance(coords, dict):
                    x = coords.get("x", 0)
                    y = coords.get("y", 0)
                else:
                    logging.error(f"Invalid coordinate format: {coords}")
                    return {"success": False, "error": "Invalid coordinate format", "verified": False}

                # If coordinates are invalid, try pressing Enter instead (for search results)
                if x <= 10 or y <= 10:
                    logging.info("Invalid coordinates, pressing Enter instead")
                    success = self.press_key("enter")
                    return {"success": success, "error": None if success else "Enter key press failed", "verified": success}
                success = self.click(x, y)
                return {"success": success, "error": None if success else "Click failed", "verified": success}
            elif action_type == "right_click":
                coords = action.get("coordinates", [0, 0])
                logging.info(f"Right-click action received with coordinates: {coords}")

                # Handle different coordinate formats
                if isinstance(coords, list) and len(coords) >= 2:
                    x, y = coords[0], coords[1]
                elif isinstance(coords, dict):
                    x = coords.get("x", 0)
                    y = coords.get("y", 0)
                else:
                    logging.error(f"Invalid coordinate format: {coords}")
                    return {"success": False, "error": "Invalid coordinate format", "verified": False}

                # If coordinates are invalid, skip the action
                if x <= 10 or y <= 10:
                    logging.info("Invalid coordinates for right-click, skipping")
                    return {"success": False, "error": "Invalid coordinates", "verified": False}
                success = self.click(x, y, button="right")
                return {"success": success, "error": None if success else "Right-click failed", "verified": success}
            elif action_type == "type":
                text = action.get("text", "")
                success = self.type_text(text)
                return {"success": success, "error": None if success else "Typing failed", "verified": success}
            elif action_type == "key_press":
                key = action.get("key", "")
                success = self.press_key(key)
                return {"success": success, "error": None if success else "Key press failed", "verified": success}
            elif action_type == "scroll":
                clicks = action.get("clicks", 1)
                coords = action.get("coordinates")
                if coords:
                    if isinstance(coords, list) and len(coords) >= 2:
                        success = self.scroll(clicks, coords[0], coords[1])
                    elif isinstance(coords, dict):
                        success = self.scroll(clicks, coords.get("x"), coords.get("y"))
                    else:
                        success = self.scroll(clicks)
                else:
                    success = self.scroll(clicks)
                return {"success": success, "error": None if success else "Scroll failed", "verified": success}
            elif action_type == "wait":
                # Skip all waiting - no delays
                logging.info("Skipping wait action - no delays enabled")
                return {"success": True, "error": None, "verified": True}
            else:
                logging.error(f"Unknown action type: {action_type}")
                return {"success": False, "error": f"Unknown action type: {action_type}", "verified": False}
        except Exception as e:
            logging.error(f"Error executing action {action_type}: {e}")
            return {"success": False, "error": str(e), "verified": False}
    
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
    
    def _take_verification_screenshot(self) -> Optional[Any]:
        """Take a screenshot for verification purposes."""
        if not pyautogui:
            return None
        try:
            return pyautogui.screenshot()
        except Exception as e:
            logging.error(f"Error taking verification screenshot: {e}")
            return None
    
    def _verify_with_gemini(self, before_image: Any, after_image: Any, action: Dict[str, Any], task_description: str) -> Dict[str, Any]:
        """Use Gemini to analyze what changed between screenshots and verify the action."""
        try:
            # Import here to avoid circular imports
            from llm.gemini.client import GeminiClient
            
            client = GeminiClient()
            
            # Create a prompt for Gemini to analyze the change
            action_desc = action.get('description', 'Unknown action')
            prompt = f"""
You are verifying if a desktop automation action was successful.

TASK: {task_description}
ACTION PERFORMED: {action_desc}

Compare the BEFORE (left) and AFTER (right) screenshots to determine:
1. What changed on screen?
2. Did the action achieve its intended purpose?
3. Was the task goal accomplished?

Respond with ONLY valid JSON:
{{
    "verified": true/false,
    "reason": "Brief explanation of what happened",
    "task_completed": true/false,
    "confidence": 0.0-1.0
}}
"""
            
            # Create a combined image showing before/after
            if before_image and after_image:
                # Create side-by-side comparison
                from PIL import Image
                import io
                
                # Resize images to same height for comparison
                height = min(before_image.height, after_image.height)
                before_resized = before_image.resize((int(before_image.width * height / before_image.height), height))
                after_resized = after_image.resize((int(after_image.width * height / after_image.height), height))
                
                # Create combined image
                combined_width = before_resized.width + after_resized.width
                combined = Image.new('RGB', (combined_width, height))
                combined.paste(before_resized, (0, 0))
                combined.paste(after_resized, (before_resized.width, 0))
                
                # Add labels
                from PIL import ImageDraw, ImageFont
                draw = ImageDraw.Draw(combined)
                try:
                    font = ImageFont.truetype("arial.ttf", 20)
                except:
                    font = ImageFont.load_default()
                
                draw.text((10, 10), "BEFORE", fill="red", font=font)
                draw.text((before_resized.width + 10, 10), "AFTER", fill="green", font=font)
                
                analysis = client.analyze_screenshot(combined, prompt)
                
                if analysis and "verified" in analysis:
                    return {
                        "verified": analysis.get("verified", False),
                        "reason": analysis.get("reason", "Gemini analysis"),
                        "task_completed": analysis.get("task_completed", False),
                        "confidence": analysis.get("confidence", 0.5)
                    }
            
        except Exception as e:
            logging.error(f"Gemini verification failed: {e}")
        
        # Fallback to basic screenshot comparison
        return self._compare_screenshots_basic(before_image, after_image)
    
    def _analyze_action_failure(self, action: Dict[str, Any], task_description: str, before_image: Any, after_image: Any) -> Dict[str, Any]:
        """Use Gemini to analyze why an action failed and suggest fixes."""
        try:
            from llm.gemini.client import GeminiClient
            
            client = GeminiClient()
            
            action_desc = action.get('description', 'Unknown action')
            action_type = action.get('action_type', 'unknown')
            
            prompt = f"""
You are a desktop automation expert analyzing why an action failed.

TASK: {task_description}
FAILED ACTION: {action_desc}
ACTION TYPE: {action_type}

The action was attempted but verification shows it did not achieve the expected result.

Analyze the BEFORE and AFTER screenshots (side-by-side) and explain:
1. What was supposed to happen?
2. What actually happened instead?
3. Why did it fail? (wrong coordinates, wrong element, timing issue, etc.)
4. What should be done differently?

Respond with ONLY valid JSON:
{{
    "expected_outcome": "What should have happened",
    "actual_outcome": "What happened instead", 
    "failure_reason": "Root cause of the failure",
    "suggested_fix": "How to fix this action",
    "alternative_approach": "Different way to accomplish the task",
    "confidence": 0.0-1.0
}}
"""
            
            # Create combined image for analysis
            from PIL import Image
            import io
            
            height = min(before_image.height, after_image.height)
            before_resized = before_image.resize((int(before_image.width * height / before_image.height), height))
            after_resized = after_image.resize((int(after_image.width * height / after_image.height), height))
            
            combined_width = before_resized.width + after_resized.width
            combined = Image.new('RGB', (combined_width, height))
            combined.paste(before_resized, (0, 0))
            combined.paste(after_resized, (before_resized.width, 0))
            
            # Add labels
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(combined)
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()
            
            draw.text((10, 10), "BEFORE (Failed Action)", fill="red", font=font)
            draw.text((before_resized.width + 10, 10), "AFTER (Result)", fill="blue", font=font)
            
            analysis = client.analyze_screenshot(combined, prompt)
            
            if analysis and "failure_reason" in analysis:
                return {
                    "analyzed": True,
                    "expected_outcome": analysis.get("expected_outcome", "Unknown"),
                    "actual_outcome": analysis.get("actual_outcome", "Unknown"),
                    "failure_reason": analysis.get("failure_reason", "Unknown"),
                    "suggested_fix": analysis.get("suggested_fix", "Unknown"),
                    "alternative_approach": analysis.get("alternative_approach", "Unknown"),
                    "confidence": analysis.get("confidence", 0.5)
                }
            
        except Exception as e:
            logging.error(f"Gemini failure analysis failed: {e}")
        
        return {
            "analyzed": False,
            "failure_reason": "Could not analyze failure",
            "suggested_fix": "Manual review required"
        }
        """Basic screenshot comparison as fallback."""
        if not before or not after:
            return {"verified": False, "reason": "Missing screenshots"}
        
        try:
            import numpy as np
            
            before_array = np.array(before)
            after_array = np.array(after)
            diff = np.abs(before_array.astype(np.float32) - after_array.astype(np.float32))
            mean_diff = np.mean(diff) / 255.0
            
            changed = mean_diff > threshold
            return {
                "verified": changed,
                "reason": f"UI changed by {mean_diff:.1%}" if changed else f"No significant UI change ({mean_diff:.1%})",
                "task_completed": changed,  # Assume task completed if UI changed
                "confidence": min(mean_diff * 100, 0.9)  # Higher change = higher confidence
            }
        except Exception as e:
            return {"verified": True, "reason": f"Comparison failed: {e}", "task_completed": True, "confidence": 0.5}
    
    def _verify_click_action(self, action: Dict[str, Any], before_screenshot: Any) -> Dict[str, Any]:
        """Verify that a click action had the expected effect."""
        coords = action.get("coordinates", [0, 0])
        if isinstance(coords, list) and len(coords) >= 2:
            x, y = coords[0], coords[1]
        else:
            return {"verified": False, "reason": "Invalid coordinates"}
        
        # Wait a moment for UI to respond
        time.sleep(0.5)
        
        after_screenshot = self._take_verification_screenshot()
        ui_changed = self._compare_screenshots(before_screenshot, after_screenshot)
        
        if ui_changed:
            return {"verified": True, "reason": "UI changed after click"}
        else:
            return {"verified": False, "reason": "No UI change detected after click"}
    
    def _verify_type_action(self, action: Dict[str, Any], before_screenshot: Any) -> Dict[str, Any]:
        """Verify that a type action had the expected effect."""
        # For typing, we expect some UI change (cursor movement, text appearing, etc.)
        time.sleep(0.3)
        
        after_screenshot = self._take_verification_screenshot()
        ui_changed = self._compare_screenshots(before_screenshot, after_screenshot)
        
        if ui_changed:
            return {"verified": True, "reason": "UI changed after typing"}
        else:
            return {"verified": False, "reason": "No UI change detected after typing"}
    
    def _verify_key_press_action(self, action: Dict[str, Any], before_screenshot: Any) -> Dict[str, Any]:
        """Verify that a key press action had the expected effect."""
        # For key presses, we expect some UI change
        time.sleep(0.3)
        
        after_screenshot = self._take_verification_screenshot()
        ui_changed = self._compare_screenshots(before_screenshot, after_screenshot)
        
        if ui_changed:
            return {"verified": True, "reason": "UI changed after key press"}
        else:
            return {"verified": False, "reason": "No UI change detected after key press"}
    
    def _verify_right_click_action(self, action: Dict[str, Any], before_screenshot: Any) -> Dict[str, Any]:
        """Verify that a right-click action opened a context menu."""
        # Right-clicks should typically open context menus, which change the UI
        time.sleep(0.5)
        
        after_screenshot = self._take_verification_screenshot()
        ui_changed = self._compare_screenshots(before_screenshot, after_screenshot)
        
        if ui_changed:
            return {"verified": True, "reason": "Context menu appeared after right-click"}
        else:
            return {"verified": False, "reason": "No context menu detected after right-click"}
    
    def _verify_task_specific_action(self, action: Dict[str, Any], task_description: str, before_screenshot: Any) -> Dict[str, Any]:
        """Perform task-specific verification based on the overall task goal."""
        # Wait for UI to update
        time.sleep(0.5)
        after_screenshot = self._take_verification_screenshot()
        
        # Use Gemini to analyze the change
        verification = self._verify_with_gemini(before_screenshot, after_screenshot, action, task_description)
        
        # If verification failed, analyze why
        if not verification.get("verified", False):
            logging.info("Action verification failed, analyzing root cause...")
            failure_analysis = self._analyze_action_failure(action, task_description, before_screenshot, after_screenshot)
            verification["failure_analysis"] = failure_analysis
            verification["reason"] += f" | Analysis: {failure_analysis.get('failure_reason', 'Unknown')}"
        
        return verification
    
    def _verify_app_launch(self, task_description: str, before_screenshot: Any) -> Dict[str, Any]:
        """Verify that an application was launched by checking for new windows."""
        time.sleep(1)  # Give time for app to launch
        
        after_screenshot = self._take_verification_screenshot()
        ui_changed = self._compare_screenshots(before_screenshot, after_screenshot)
        
        if ui_changed:
            return {"verified": True, "reason": "Application window appeared"}
        else:
            return {"verified": False, "reason": "No new application window detected"}
    
    def _verify_text_entry(self, action: Dict[str, Any], before_screenshot: Any) -> Dict[str, Any]:
        """Verify that text was entered by checking for UI changes."""
        # For text entry, we expect some visual change (cursor movement, text appearance, etc.)
        time.sleep(0.3)
        
        after_screenshot = self._take_verification_screenshot()
        ui_changed = self._compare_screenshots(before_screenshot, after_screenshot)
        
        expected_text = action.get("text", "")
        if ui_changed:
            return {"verified": True, "reason": f"Text '{expected_text[:20]}...' appeared in UI"}
        else:
            return {"verified": False, "reason": f"Text '{expected_text[:20]}...' not detected in UI"}
    
    def verify_action_execution(self, action: Dict[str, Any], task_description: str = "") -> Dict[str, Any]:
        """Verify that an action was executed successfully and contributed to task completion."""
        action_type = action.get("action_type", "").lower()
        
        # Take screenshot before action for comparison
        before_screenshot = self._take_verification_screenshot()
        
        # Execute the action first using the individual methods
        try:
            if action_type == "click":
                coords = action.get("coordinates", [0, 0])
                if isinstance(coords, list) and len(coords) >= 2:
                    x, y = coords[0], coords[1]
                elif isinstance(coords, dict):
                    x = coords.get("x", 0)
                    y = coords.get("y", 0)
                else:
                    return {"success": False, "error": "Invalid coordinate format", "verified": False}
                
                if x <= 10 or y <= 10:
                    success = self.press_key("enter")
                else:
                    success = self.click(x, y)
            elif action_type == "right_click":
                coords = action.get("coordinates", [0, 0])
                if isinstance(coords, list) and len(coords) >= 2:
                    x, y = coords[0], coords[1]
                elif isinstance(coords, dict):
                    x = coords.get("x", 0)
                    y = coords.get("y", 0)
                else:
                    return {"success": False, "error": "Invalid coordinate format", "verified": False}
                
                if x <= 10 or y <= 10:
                    success = False
                else:
                    success = self.click(x, y, button="right")
            elif action_type == "type":
                text = action.get("text", "")
                success = self.type_text(text)
            elif action_type == "key_press":
                key = action.get("key", "")
                success = self.press_key(key)
            elif action_type == "scroll":
                clicks = action.get("clicks", 1)
                coords = action.get("coordinates")
                if coords:
                    if isinstance(coords, list) and len(coords) >= 2:
                        success = self.scroll(clicks, coords[0], coords[1])
                    elif isinstance(coords, dict):
                        success = self.scroll(clicks, coords.get("x"), coords.get("y"))
                    else:
                        success = self.scroll(clicks)
                else:
                    success = self.scroll(clicks)
            elif action_type == "wait":
                success = True
            else:
                return {"success": False, "error": f"Unknown action type: {action_type}", "verified": False}
        except Exception as e:
            return {"success": False, "error": str(e), "verified": False}
        
        if not success:
            return {
                "success": False,
                "error": "Action execution failed",
                "verified": False
            }
        
        # Perform task-specific verification
        verification = self._verify_task_specific_action(action, task_description, before_screenshot)
        
        verification["success"] = success
        verification["error"] = None
        return verification
