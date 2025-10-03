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
    from llm.gemini_client import GeminiClient
except ImportError:
    GeminiClient = None

try:
    from interfaces.voice import VoiceHandler
except ImportError:
    VoiceHandler = None

try:
    from automation.screen import ScreenAnalyzer
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
        
        # Initialize components
        if ScreenAnalyzer:
            self.screen_analyzer = ScreenAnalyzer(self.config.screenshot_dir)
        else:
            self.screen_analyzer = None
            
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
    
    def execute_task(self, task_description: str) -> Dict[str, Any]:
        """Execute a high-level task with multi-step capability."""
        logging.info(f"Starting task: {task_description}")
        
        self.current_task = task_description
        self.task_progress = []
        self.is_running = True
        
        # Check if this is a multi-step task
        multi_step_indicators = [
            'and', 'then', 'check for', 'download', 'install', 'update', 
            'click on', 'navigate to', 'find', 'search for', 'select'
        ]
        is_multi_step = any(indicator in task_description.lower() for indicator in multi_step_indicators)
        
        if is_multi_step:
            logging.info("Detected multi-step task - will use iterative approach")
            return self._execute_multi_step_task(task_description)
        else:
            logging.info("Single-step task - using standard approach")
            return self._execute_single_step_task(task_description)
    
    def _execute_multi_step_task(self, task_description: str) -> Dict[str, Any]:
        """Execute a complex task that may require multiple screenshots and analysis cycles."""
        max_iterations = 5
        all_results = []
        original_task = task_description
        
        try:
            for iteration in range(max_iterations):
                logging.info(f"Multi-step iteration {iteration + 1}/{max_iterations}")
                
                # Take screenshot of current state
                screenshot = self.screen_analyzer.capture_screenshot()
                
                # Create context-aware prompt that includes what we've accomplished
                if iteration == 0:
                    current_prompt = f"TASK: {task_description}\n\nThis is the beginning of a multi-step task. Analyze what needs to be done first."
                else:
                    completed_actions = [r.get('description', 'Unknown action') for results in all_results for r in results if r.get('success')]
                    current_prompt = f"ORIGINAL TASK: {original_task}\n\nCOMPLETED STEPS: {completed_actions}\n\nCurrent screen shows the result of previous actions. What should be done next to complete the original task? If the task appears complete, respond with no actions."
                
                # Get AI analysis for current step
                analysis = self.gemini_client.analyze_screenshot(screenshot, current_prompt)
                
                if "error" in analysis:
                    logging.error(f"Analysis error: {analysis['error']}")
                    break
                
                actions = analysis.get("actions", [])
                # If heuristic fallback used on later iteration, prune duplicates
                if analysis.get('fallback_used') and iteration > 0 and actions:
                    before_ct = len(actions)
                    actions = self._prune_redundant_fallback(actions)
                    if len(actions) != before_ct:
                        logging.info(f"Pruned {before_ct - len(actions)} duplicate fallback actions")
                    if not actions:
                        # Try infer next logical step
                        inferred = self._infer_next_step(original_task)
                        if inferred:
                            actions = [inferred]
                            logging.info(f"Injected inferred action: {inferred.get('description')}")
                        else:
                            logging.info("No further inferred actions; ending multi-step process")
                            break
                
                # Check if task appears complete (no more actions needed)
                if not actions:
                    logging.info("Task appears complete - no more actions suggested")
                    break
                
                logging.info(f"AI generated {len(actions)} actions for iteration {iteration + 1}: {[a.get('description', 'No desc') for a in actions]}")
                
                # Execute this iteration's actions
                results = self._execute_action_sequence(actions)
                all_results.append(results)

                # If this batch came from fallback planner, stop further iterations
                if analysis.get('fallback_used'):
                    logging.info("Heuristic fallback plan executed; ending multi-step iterations early.")
                    break
                
                # Check if all actions failed - might indicate we're stuck
                if all(not r.get("success", False) for r in results):
                    logging.warning("All actions failed - ending multi-step process")
                    break
                
                # Brief pause between iterations to let UI update
                import time
                time.sleep(1)
            
            # Calculate overall success
            total_successful = sum(len([r for r in results if r.get("success", False)]) for results in all_results)
            total_actions = sum(len(results) for results in all_results)
            
            return {
                "success": total_successful > 0,
                "task_description": original_task,
                "initial_analysis": f"Multi-step task completed in {len(all_results)} iterations",
                "actions_completed": total_successful,
                "total_actions": total_actions,
                "action_results": [item for sublist in all_results for item in sublist],
                "iterations": len(all_results)
            }
            
        except Exception as e:
            logging.error(f"Error in multi-step task: {e}")
            return {
                "success": False,
                "error": str(e),
                "actions_completed": len(self.task_progress)
            }
        finally:
            self.is_running = False

    def _prune_redundant_fallback(self, actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        existing = {
            (r.get('action', {}).get('description','').lower())
            for r in self.task_progress if r.get('success')
        }
        return [a for a in actions if a.get('description','').lower() not in existing]

    def _infer_next_step(self, original_task: str) -> Optional[Dict[str, Any]]:
        t = original_task.lower()
        if any(k in t for k in ['youtube','play','video','song']):
            # If search submitted but no video clicked, click first video
            submitted = any(
                ('submit search' in (r.get('action', {}).get('description','').lower()) or
                 'submit youtube search' in (r.get('action', {}).get('description','').lower())) and r.get('success')
                for r in self.task_progress
            )
            clicked_video = any('click first youtube video' in (r.get('action', {}).get('description','').lower()) for r in self.task_progress)
            if submitted and not clicked_video:
                try:
                    if pyautogui:
                        screen_w, screen_h = pyautogui.size()
                    else:
                        screen_w, screen_h = 1920, 1080
                except Exception:
                    screen_w, screen_h = 1920,1080
                return {
                    'action_type':'click',
                    'coordinates':[int(screen_w*0.25), int(screen_h*0.35)],
                    'description':'Click first YouTube video result (inferred)'
                }
        return None
    
    def _execute_single_step_task(self, task_description: str) -> Dict[str, Any]:
        """Execute a simple single-step task (original behavior)."""
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
