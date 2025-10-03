#!/usr/bin/env python3
"""
Gemini Client Module for Truvo Desktop Assistant
Client for interacting with Google's Gemini AI API.
"""

import os
import sys
import json
import logging
import contextlib
from typing import Optional, Dict, Any, List
from PIL import Image
from dataclasses import dataclass

# Suppress stderr temporarily
@contextlib.contextmanager
def suppress_stderr():
    """Suppress stderr temporarily."""
    with open(os.devnull, 'w') as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr

# Google AI imports with error suppression
with suppress_stderr():
    import google.generativeai as genai

# Minimal config fallback (avoid circular import with main.py)
@dataclass
class AssistantConfig:
    """Configuration for Gemini client (lightweight version)."""
    model_name: str = "auto"
    offline_fallback_enabled: bool = True

# Try to import OS detection
try:
    from utils.os_detection import get_os_context, get_os_commands, os_detector
    OS_DETECTION_AVAILABLE = True
except ImportError:
    get_os_context = None
    get_os_commands = None
    OS_DETECTION_AVAILABLE = False
    os_detector = None


class GeminiClient:
    """Client for interacting with Google's Gemini AI API."""
    PREFERRED_MODELS = [
        # Prioritize models with highest quota limits and best availability
        'gemini-1.5-flash',        # Best quota: 1500 requests/day, most stable
        'gemini-1.5-flash-8b',     # Good quota, lightweight, fast
        'gemini-1.5-pro',          # 50 requests/day but very stable
        'gemini-2.5-flash',        # Latest with good availability
        'gemini-2.0-flash-exp',    # Experimental fallback
        'gemini-2.0-flash',        # Lower quota: 200 requests/day, avoid if possible
    ]

    def __init__(self, api_key: Optional[str] = None, config: Optional[AssistantConfig] = None):
        self.api_key = api_key or os.getenv('GOOGLE_AI_API_KEY')
        if not self.api_key:
            raise ValueError("Google AI API key is required")
        self.config = config or AssistantConfig()
        with suppress_stderr():
            genai.configure(api_key=self.api_key)
        chosen = self._select_model_with_fallback(self.config.model_name)
        logging.info(f"Using Gemini model: {chosen}")
        with suppress_stderr():
            self.model = genai.GenerativeModel(chosen)

    def _select_model_with_fallback(self, requested: str) -> str:
        """Select model with intelligent fallback and quota awareness."""
        try:
            models = list(genai.list_models())
            model_names = {m.name.split('/')[-1]: m for m in models}
        except Exception as e:
            logging.warning(f"Could not list models ({e}); using fallback")
            return 'gemini-1.5-flash'  # Most reliable fallback

        # If specific model requested and available, use it
        if requested != 'auto' and requested in model_names:
            return requested

        # Auto selection with quota-aware prioritization
        if requested == 'auto':
            print("Testing model availability...")
            # Try each preferred model with feedback
            for candidate in self.PREFERRED_MODELS:
                if candidate in model_names:
                    print(f"  Testing {candidate}...")
                    if self._test_model_availability(candidate):
                        print(f"Selected {candidate} (available and working)")
                        return candidate
            
            # If all preferred models fail, try any available model
            print("Trying alternative models...")
            for name in sorted(model_names.keys()):
                if any(keyword in name for keyword in ['flash', 'pro']) and name not in self.PREFERRED_MODELS:
                    print(f"  Testing {name}...")
                    if self._test_model_availability(name):
                        print(f"Found working alternative: {name}")
                        return name
        
        # Final fallback to most reliable models
        print("All preferred models unavailable. Trying final fallbacks...")
        fallback_models = ['gemini-1.5-flash', 'gemini-1.5-flash-8b', 'gemini-pro']
        
        for fallback in fallback_models:
            if fallback in model_names:
                print(f"  Testing {fallback}...")
                if self._test_model_availability(fallback):
                    return fallback
        
        # Last resort - return the most stable model even if untested
        print("Warning: Using untested fallback model (quota may be exceeded)")
        return 'gemini-1.5-flash'
    
    def _test_model_availability(self, model_name: str) -> bool:
        """Test if a model is currently available (not hitting quota limits)."""
        try:
            # Create a temporary model instance to test
            test_model = genai.GenerativeModel(model_name)
            
            # Try a simple generation request with minimal tokens
            with suppress_stderr():
                response = test_model.generate_content("Hi", 
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=1,
                        temperature=0
                    ))
                # If we get any response, model is working
                return response is not None
            
        except Exception as e:
            error_msg = str(e).lower()
            # Check for quota/rate limit errors
            if any(keyword in error_msg for keyword in ['429', 'quota', 'rate limit', 'requests per']):
                print(f"  {model_name}: Quota exceeded")
                return False
            # Check for model unavailable errors
            elif any(keyword in error_msg for keyword in ['not found', 'unavailable', 'invalid']):
                print(f"  {model_name}: Not available")
                return False
            else:
                # Other errors might be temporary or authentication issues
                print(f"  {model_name}: Error - {str(e)[:50]}...")
                return False
        
        return False

    def _select_model(self, requested: str) -> str:
        """Legacy method - kept for compatibility."""
        return self._select_model_with_fallback(requested)

    @staticmethod
    def list_available_models() -> List[str]:
        try:
            # Only include generative models (filter out embedding, imagen, veo, etc.)
            generative_keywords = ['flash', 'pro']
            return [
                m.name.split('/')[-1]
                for m in genai.list_models()
                if any(kw in m.name.split('/')[-1] for kw in generative_keywords)
            ]
        except Exception as e:
            logging.error(f"Failed to list models: {e}")
            return []
        
    def analyze_screenshot(self, image: Image.Image, task_description: str) -> Dict[str, Any]:
        """Analyze a screenshot and provide action recommendations."""
        try:
            # Get OS-specific context and commands
            os_context = ""
            os_commands = {}
            if OS_DETECTION_AVAILABLE and get_os_context and get_os_commands and os_detector:
                try:
                    os_context = f"\nSYSTEM ENVIRONMENT: {get_os_context()}\n"
                    os_commands = get_os_commands()
                    
                    # Add OS-specific instructions
                    if os_detector.os_info['is_windows']:
                        os_context += "- Use Windows UI elements: Start menu, taskbar, Windows applications\n"
                        os_context += "- File paths use backslashes (\\) \n"
                        os_context += "- Key combinations like Ctrl+L must be written as 'ctrl+l'\n"
                    elif os_detector.os_info['is_macos']:
                        os_context += "- Use macOS UI elements: Dock, Launchpad, Applications folder\n"
                        os_context += "- File paths use forward slashes (/)\n"
                        os_context += "- Key combinations use Cmd instead of Ctrl (e.g., 'cmd+l')\n"
                    elif os_detector.os_info['is_linux']:
                        os_context += "- Use Linux desktop environment elements\n"
                        os_context += "- File paths use forward slashes (/)\n"
                        os_context += "- Key combinations typically use Ctrl\n"
                except Exception as e:
                    os_context = ""
            
            prompt = f"""
You are a desktop assistant AI that can see and understand computer screens.

TASK: {task_description}
{os_context}
CRITICAL INSTRUCTIONS:
- NEVER type terminal/command line commands
- NEVER use "edge website.com" command syntax  
- Use appropriate UI elements for this operating system
- Key combinations must be written in lowercase with + (e.g., "ctrl+l")

TASK CLASSIFICATION:
- System/App tasks (settings, updates, calculator, notepad, office apps) → Open applications directly
- Website tasks (go to website.com, visit site.org, chatgpt.com) → Open browser first, then navigate
- Browser tasks (open chrome, open edge, open firefox) → Open browser app only

For SYSTEM/APP tasks like "settings", "updates", "calculator", "notepad":
1. Press "win" key (opens Start menu)  
2. Type the app name (e.g., "settings", "calculator", "notepad")
3. Press "enter" (launches the Windows app)

For WEBSITE tasks containing URLs (.com, .org) or "go to" phrases:
1. Press "win" key (opens Start menu)
2. Type browser name (e.g., "edge", "chrome") 
3. Press "enter" (launches browser)
4. Press "ctrl+l" (focuses address bar)
5. Type website URL only
6. Press "enter" (navigates to site)

Please respond in this JSON format:
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
            "text": "settings",
            "description": "Type 'settings' to search for Settings app",
            "confidence": 0.9
        }},
        {{
            "action_type": "key_press",
            "key": "enter", 
            "description": "Press Enter to open Settings",
            "confidence": 0.9
        }}
    ],
    "safety_concerns": [],
    "next_steps": "Application should open",
    "overall_confidence": 0.9
}}
"""
            
            with suppress_stderr():
                response = self.model.generate_content([prompt, image])
            return self._parse_response(response.text)
            
        except Exception as e:
            err_text = str(e)
            logging.error(f"Error analyzing screenshot: {err_text}")
            # Quota / rate limit fallback
            if self.config.offline_fallback_enabled and ('quota' in err_text.lower() or '429' in err_text):
                logging.warning("Using offline heuristic fallback planner due to quota/rate limit")
                return self._rule_based_plan(task_description)
            return {"error": err_text, "actions": [], "confidence": 0.0}

    def _rule_based_plan(self, task: str) -> Dict[str, Any]:
        """Heuristic local planner used when API unavailable (rate limits etc.)."""
        t = task.lower()
        actions: List[Dict[str, Any]] = []
        understanding = "Heuristic fallback plan generated (no model)."
        confidence = 0.4
        # Basic primitives
        def start_and_type(app: str):
            return [
                {"action_type": "key_press", "key": "win", "description": "Open start menu (fallback)"},
                {"action_type": "type", "text": app, "description": f"Type '{app}' in start menu (fallback)"},
                {"action_type": "key_press", "key": "enter", "description": f"Launch {app} (fallback)"},
            ]
        if any(k in t for k in ["youtube", "song", "video", "play"]):
            # Extract a probable query after 'search' or 'for'
            import re
            query = ""
            m = re.search(r'search(?: for)? (.*)', t)
            if m:
                query = m.group(1)
            query = query.replace('play', '').replace('youtube', '').strip()
            actions.extend(start_and_type('edge'))
            actions.append({"action_type": "key_press", "key": "ctrl+l", "description": "Focus address bar (fallback)"})
            actions.append({"action_type": "type", "text": "https://www.youtube.com", "description": "Type YouTube URL (fallback)"})
            actions.append({"action_type": "key_press", "key": "enter", "description": "Navigate to YouTube (fallback)"})
            if query:
                actions.append({"action_type": "click", "description": "Click YouTube search bar (fallback)"})
                actions.append({"action_type": "type", "text": query, "description": f"Type search query '{query}' (fallback)"})
                actions.append({"action_type": "key_press", "key": "enter", "description": "Submit search (fallback)"})
        elif any(k in t for k in ["update", "windows update", "check for updates"]):
            actions.extend(start_and_type('settings'))
            actions.append({"action_type": "type", "text": "windows update", "description": "Type 'windows update' inside Settings (fallback)"})
            actions.append({"action_type": "key_press", "key": "enter", "description": "Open Windows Update section (fallback)"})
        elif any(k in t for k in ["notepad", "note pad"]):
            actions.extend(start_and_type('notepad'))
        elif any(k in t for k in ["calc", "calculator"]):
            actions.extend(start_and_type('calculator'))
        else:
            # Generic attempt: open start and type something meaningful
            words = t.split()
            if words:
                actions.extend(start_and_type(words[0]))
        return {
            "understanding": understanding,
            "actions": actions,
            "safety_concerns": [],
            "next_steps": "Executed heuristic plan; consider re-running when quota resets",
            "overall_confidence": confidence,
            "fallback_used": True
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
