#!/usr/bin/env python3
"""
Truvo - AI-Powered Desktop Assistant

A self-operating desktop assistant powered by Google's Gemini AI that can see,
understand, and interact with your desktop environment through natural language commands.
"""

# Suppress Google AI library warnings
import os
import warnings
import logging as default_logging

# Multiple layers of Google library warning suppression
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '3'  # More aggressive suppression
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GRPC_TRACE'] = ''
os.environ['GRPC_VERBOSITY'] = 'NONE'

# Suppress all warnings
warnings.filterwarnings('ignore')
default_logging.getLogger('absl').setLevel(default_logging.ERROR)

import sys
import time
import json
import base64
import logging
import argparse
import contextlib
from io import BytesIO
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
from dataclasses import dataclass

# Global variable for selected voice
SELECTED_VOICE_ID = None

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

# Third-party imports
import pyautogui
import cv2
import numpy as np
from PIL import Image, ImageDraw
from pynput import mouse, keyboard
from pynput.keyboard import Key
from dotenv import load_dotenv

# Google AI imports with error suppression
with suppress_stderr():
    import google.generativeai as genai

# NLP and ML imports
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: NLP libraries not installed. Using basic keyword detection. Install with: pip install scikit-learn nltk")

# Voice recognition imports
try:
    import speech_recognition as sr
    import pyttsx3
    import threading
    import queue
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False
    print("Warning: Voice libraries not installed. Voice mode disabled. Install with: pip install SpeechRecognition pyttsx3 pyaudio")

# OS Detection import
try:
    from os_detection import get_os_context, get_os_commands, os_detector
    OS_DETECTION_AVAILABLE = True
except ImportError:
    OS_DETECTION_AVAILABLE = False
    print("Warning: OS detection not available")


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
    offline_fallback_enabled: bool = True  # Use heuristic planner if API quota exceeded
    model_name: str = "auto"  # 'auto' selects best available
    # Voice settings
    voice_enabled: bool = False  # Voice mode disabled by default
    voice_language: str = "en-US"  # Default language for speech recognition
    tts_rate: int = 150  # Text-to-speech speech rate (words per minute)
    tts_volume: float = 0.9  # Text-to-speech volume (0.0 to 1.0)
    voice_timeout: float = 5.0  # Seconds to wait for speech input
    push_to_talk_key: str = "space"  # Key for push-to-talk mode


# =============================================================================
# NLP Intent Classification
# =============================================================================

class IntentClassifier:
    """NLP-based intent classifier to distinguish between chat and automation tasks."""
    
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.is_trained = False
        
        if SKLEARN_AVAILABLE:
            self._initialize_nltk()
            self._train_classifier()
        else:
            print("🔄 Using fallback keyword-based detection (install scikit-learn for better accuracy)")
    
    def _initialize_nltk(self):
        """Initialize NLTK resources."""
        try:
            import ssl
            try:
                _create_unverified_https_context = ssl._create_unverified_context
            except AttributeError:
                pass
            else:
                ssl._create_default_https_context = _create_unverified_https_context
            
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
        except Exception as e:
            print(f"Warning: NLTK setup warning: {e}")
    
    def _preprocess_text(self, text):
        """Preprocess text for classification."""
        try:
            # Tokenize and clean
            tokens = word_tokenize(text.lower())
            
            # Remove stopwords
            stop_words = set(stopwords.words('english'))
            filtered_tokens = [w for w in tokens if w not in stop_words and w.isalpha()]
            
            # Lemmatize
            lemmatizer = WordNetLemmatizer()
            lemmatized = [lemmatizer.lemmatize(token) for token in filtered_tokens]
            
            return ' '.join(lemmatized)
        except Exception:
            # Fallback to simple preprocessing
            return text.lower().strip()
    
    def _train_classifier(self):
        """Train the intent classification model."""
        try:
            # Training data: (text, label) where 0=automation, 1=conversation
            training_data = [
                # Automation tasks (0)
                ("open chrome browser", 0),
                ("click on the button", 0),
                ("type hello world", 0),
                ("press enter key", 0),
                ("scroll down page", 0),
                ("launch notepad application", 0),
                ("close window", 0),
                ("minimize application", 0),
                ("take screenshot", 0),
                ("open file explorer", 0),
                ("start calculator", 0),
                ("run program", 0),
                ("navigate to website", 0),
                ("download file", 0),
                ("save document", 0),
                ("copy text", 0),
                ("paste clipboard", 0),
                ("delete file", 0),
                ("install software", 0),
                ("update application", 0),
                ("switch to tab", 0),
                ("maximize window", 0),
                ("drag and drop", 0),
                ("select all text", 0),
                ("go to url", 0),
                ("open edge", 0),
                ("start spotify", 0),
                ("launch terminal", 0),
                ("execute command", 0),
                ("find text", 0),
                ("can you open edge", 0),
                ("can you open chrome", 0),
                ("open edge for me", 0),
                ("like can you open edge", 0),
                ("please open chrome", 0),
                ("launch notepad please", 0),
                ("start calculator for me", 0),
                ("can you click the button", 0),
                ("please type this text", 0),
                ("can you scroll down", 0),
                ("help me open firefox", 0),
                ("open the browser", 0),
                ("start the application", 0),
                ("run the program", 0),
                ("execute this command", 0),
                ("launch spotify app", 0),
                ("open file manager", 0),
                ("start windows explorer", 0),
                
                # Conversational queries (1)
                ("hello", 1),
                ("hi", 1),
                ("hey", 1),
                ("hello there", 1),
                ("hi there", 1),
                ("hey there", 1),
                ("good morning", 1),
                ("good afternoon", 1),
                ("good evening", 1),
                ("how are you", 1),
                ("what is your name", 1),
                ("tell me a joke", 1),
                ("how do you work", 1),
                ("what can you do", 1),
                ("explain machine learning", 1),
                ("what do you think about", 1),
                ("are you intelligent", 1),
                ("do you have feelings", 1),
                ("thanks for helping", 1),
                ("you are awesome", 1),
                ("good job well done", 1),
                ("i love you", 1),
                ("how was your day", 1),
                ("what is the weather", 1),
                ("tell me about science", 1),
                ("why is sky blue", 1),
                ("what time is it", 1),
                ("who invented computer", 1),
                ("recommend a movie", 1),
                ("i feel sad today", 1),
                ("you make me happy", 1),
                ("goodbye see you later", 1),
                ("hi there", 1),
                ("hey buddy", 1),
                ("thank you so much", 1),
                ("amazing work", 1),
                ("that was great", 1),
                ("i think you are", 1),
                ("do you believe in", 1),
                ("what is meaning of life", 1),
                ("tell me story", 1),
                ("sing a song", 1),
                ("make me laugh", 1),
                ("i need emotional support", 1),
                ("you are my friend", 1),
                ("how old are you", 1),
                ("where are you from", 1),
                ("what are your hobbies", 1),
                ("do you dream", 1),
                ("are you lonely", 1),
                ("what makes you happy", 1),
                ("hiii", 1),
                ("yeppppp", 1),
                ("yeahhh", 1),
                ("niceee", 1),
                ("coollll", 1),
                ("can you help me with something", 1),
                ("what do you think about this", 1),
                ("i want to ask you something", 1),
                ("are you there", 1),
                ("how smart are you", 1),
                ("do you understand me", 1),
                ("can you perform a task for me", 1),  # This should be conversational!
                ("what task can you do", 1),
                ("what kind of tasks", 1),
                ("can you do work for me", 1),
                ("help me with a task", 1),
                ("i need help with task", 1),
                ("what automation can you do", 1),
                ("like batman is the prime example", 1),
                ("batman sacrificed everything", 1),
                ("like in the movie", 1),
                ("like the character", 1),
                ("this reminds me of", 1),
            ]
            
            # Separate texts and labels
            texts = [self._preprocess_text(text) for text, label in training_data]
            labels = [label for text, label in training_data]
            
            # Create and train the model
            self.model = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=1000, ngram_range=(1, 2))),
                ('classifier', MultinomialNB(alpha=0.1))
            ])
            
            self.model.fit(texts, labels)
            self.is_trained = True
            
            # Train classifier silently
            accuracy = self.model.score(texts, labels)
            
        except Exception as e:
            print(f"Warning: NLP training failed: {e}. Using fallback detection.")
            self.is_trained = False
    
    def classify_intent(self, text: str) -> bool:
        """
        Classify if text is conversational (True) or automation task (False).
        Uses NLP if available, otherwise falls back to keyword detection.
        """
        if not text.strip():
            return True  # Empty text is conversational
        
        if SKLEARN_AVAILABLE and self.is_trained:
            try:
                processed_text = self._preprocess_text(text)
                prediction = self.model.predict([processed_text])[0]
                probabilities = self.model.predict_proba([processed_text])[0]
                
                # Get confidence scores
                automation_confidence = probabilities[0]
                conversation_confidence = probabilities[1]
                max_confidence = max(automation_confidence, conversation_confidence)
                
                # Apply confidence threshold - if confidence is too low, default to conversation
                CONFIDENCE_THRESHOLD = 0.7  # 70% confidence required
                
                if max_confidence < CONFIDENCE_THRESHOLD:
                    return True  # Default to conversation for ambiguous cases
                
                return prediction == 1  # 1 = conversation, 0 = automation
                
            except Exception as e:
                print(f"Warning: NLP classification failed: {e}. Using fallback.")
                return self._fallback_detection(text)
        else:
            return self._fallback_detection(text)
    
    def _fallback_detection(self, text: str) -> bool:
        """Fallback keyword-based detection when NLP is not available."""
        text_lower = text.lower().strip()
        
        # Clear automation commands
        automation_patterns = [
            'open ', 'launch ', 'start ', 'run ', 'execute ',
            'click', 'press ', 'type ', 'scroll', 'drag',
            'close ', 'minimize ', 'maximize ', 'switch to',
            'go to', 'navigate', 'download', 'save', 'delete',
            'copy', 'paste', 'install', 'screenshot'
        ]
        
        # Website/URL patterns - these are likely automation tasks
        website_patterns = [
            '.com', '.org', '.net', '.edu', '.gov', '.io',
            'google ', 'youtube', 'facebook', 'twitter', 'instagram',
            'website', 'browser', 'search ', 'www.'
        ]
        
        for pattern in automation_patterns:
            if pattern in text_lower:
                return False
                
        for pattern in website_patterns:
            if pattern in text_lower:
                return False  # Websites are automation tasks
        
        # Clear conversational patterns
        conversational_patterns = [
            text_lower in ['hi', 'hello', 'hey', 'thanks', 'bye'],
            text_lower.startswith(('what', 'how', 'why', 'who', 'when', 'where')),
            '?' in text,
            any(phrase in text_lower for phrase in [
                'tell me', 'explain', 'you are', 'i am', 'i feel',
                'love you', 'thank you', 'good job', 'awesome'
            ])
        ]
        
        return any(conversational_patterns)


# =============================================================================
# Core Classes
# =============================================================================

class GeminiClient:
    """Client for interacting with Google's Gemini AI API."""
    PREFERRED_MODELS = [
        # Prioritize models less likely to hit quota limits
        'gemini-1.5-flash',        # Stable, widely available
        'gemini-1.5-flash-8b',     # Lightweight, good for quotas  
        'gemini-1.5-pro',          # Reliable fallback
        'gemini-2.0-flash',        # Newer but more stable than exp
        'gemini-2.0-flash-exp',    # Experimental, likely to hit limits
        'gemini-2.5-flash',        # Latest but may have quota issues
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
            # Try each preferred model silently
            for candidate in self.PREFERRED_MODELS:
                if candidate in model_names:
                    if self._test_model_availability(candidate):
                        return candidate
                    else:
                        print(f"Warning: {candidate} unavailable (likely quota exceeded)")
            
            # If all preferred models fail, try any available model
            print("Trying alternative models...")
            for name in model_names:
                if any(keyword in name for keyword in ['flash', 'pro']) and name not in self.PREFERRED_MODELS:
                    if self._test_model_availability(name):
                        print(f"Found working alternative: {name}")
                        return name
        
        # Fallback to most reliable model
        print("Warning: Using fallback model (may have limited availability)")
        return 'gemini-1.5-flash'
    
    def _test_model_availability(self, model_name: str) -> bool:
        """Test if a model is currently available (not hitting quota limits)."""
        try:
            # Create a temporary model instance to test
            test_model = genai.GenerativeModel(model_name)
            
            # Try a simple generation request
            with suppress_stderr():
                response = test_model.generate_content("Hi", 
                    generation_config=genai.types.GenerationConfig(max_output_tokens=1))
            
            return True  # If we get here, model is working
            
        except Exception as e:
            error_msg = str(e).lower()
            if '429' in error_msg or 'quota' in error_msg or 'rate' in error_msg:
                return False  # Quota exceeded
            elif 'not found' in error_msg or 'unavailable' in error_msg:
                return False  # Model not available
            else:
                # Other errors might be temporary, assume model is available
                return True

    def _select_model(self, requested: str) -> str:
        """Legacy method - kept for compatibility."""
        return self._select_model_with_fallback(requested)

    @staticmethod
    def list_available_models() -> List[str]:
        try:
            return [m.name.split('/')[-1] for m in genai.list_models()]
        except Exception as e:
            logging.error(f"Failed to list models: {e}")
            return []
        
    def analyze_screenshot(self, image: Image.Image, task_description: str) -> Dict[str, Any]:
        """Analyze a screenshot and provide action recommendations."""
        try:
            # Get OS-specific context and commands
            os_context = ""
            os_commands = {}
            if OS_DETECTION_AVAILABLE:
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


class ChatBot:
    """Conversational chatbot using Gemini AI for natural language interactions."""
    
    def __init__(self, gemini_client: GeminiClient):
        self.gemini_client = gemini_client
        self.conversation_history = []
        self.max_history_length = 20  # Keep last 20 exchanges
        self.intent_classifier = IntentClassifier()  # NLP-based classifier
        
        # System prompt for the chatbot personality with OS context
        os_context = ""
        if OS_DETECTION_AVAILABLE:
            try:
                os_context = f"\n\nSYSTEM ENVIRONMENT: {get_os_context()}\n"
                os_context += "IMPORTANT: When providing commands, file paths, or technical instructions, use the appropriate format for this operating system.\n"
            except Exception as e:
                os_context = ""
        
        self.system_prompt = f"""You are a helpful and friendly AI assistant integrated into Truvo, a desktop automation tool. 
        You can have natural conversations about any topic while also being aware that you're part of a system 
        that can perform desktop automation tasks.{os_context}
        
        When users ask general questions, provide helpful and engaging responses. 
        
        When users ask about your capabilities or what tasks you can perform, explain that you can:
        1. Have natural conversations on any topic
        2. Perform desktop automation tasks like:
           - Opening applications (browsers, notepad, calculator, etc.)
           - Clicking buttons and UI elements
           - Typing text and pressing keys
           - Taking screenshots and analyzing screens
           - Navigating websites and software
           
        If someone asks "can you perform a task for me?" or similar, respond conversationally by asking 
        what specific task they have in mind, rather than immediately trying to automate something.
        
        Be friendly, helpful, and engaging in all interactions."""
        
    def chat(self, user_message: str) -> str:
        """Process a conversational message and return a response."""
        try:
            # Add user message to history
            self.conversation_history.append({
                "role": "user",
                "content": user_message,
                "timestamp": datetime.now().isoformat()
            })
            
            # Build conversation context
            context = self._build_conversation_context()
            
            # Generate response using Gemini
            with suppress_stderr():
                response = self.gemini_client.model.generate_content(context)
            
            if response and response.text:
                bot_response = response.text.strip()
                
                # Add bot response to history
                self.conversation_history.append({
                    "role": "assistant", 
                    "content": bot_response,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Trim history if too long
                self._trim_history()
                
                return bot_response
            else:
                return "I'm sorry, I couldn't generate a response right now. Please try again."
                
        except Exception as e:
            logging.error(f"Chat error: {e}")
            return f"I encountered an error: {str(e)}"
    
    def _build_conversation_context(self) -> str:
        """Build the conversation context for Gemini."""
        context_parts = [self.system_prompt, "\n\nConversation History:"]
        
        # Add recent conversation history
        for entry in self.conversation_history[-10:]:  # Last 10 exchanges
            role = "Human" if entry["role"] == "user" else "Assistant"
            context_parts.append(f"{role}: {entry['content']}")
        
        # Add current conversation marker
        if self.conversation_history:
            context_parts.append("\nAssistant:")
        
        return "\n".join(context_parts)
    
    def _trim_history(self):
        """Keep conversation history within limits."""
        if len(self.conversation_history) > self.max_history_length * 2:  # *2 for user+assistant pairs
            # Remove oldest pairs, keeping system prompt effectiveness
            self.conversation_history = self.conversation_history[-(self.max_history_length * 2):]
    
    def clear_history(self):
        """Clear the conversation history."""
        self.conversation_history.clear()
        
    def get_history_summary(self) -> str:
        """Get a summary of conversation history."""
        if not self.conversation_history:
            return "No conversation history yet."
            
        total_messages = len(self.conversation_history)
        user_messages = len([msg for msg in self.conversation_history if msg["role"] == "user"])
        
        if self.conversation_history:
            first_msg_time = datetime.fromisoformat(self.conversation_history[0]["timestamp"])
            last_msg_time = datetime.fromisoformat(self.conversation_history[-1]["timestamp"])
            duration = last_msg_time - first_msg_time
            
            return f"Conversation: {user_messages} exchanges, {total_messages} total messages, " \
                   f"duration: {duration.total_seconds():.0f}s"
        
        return f"Conversation: {user_messages} exchanges, {total_messages} total messages"
    
    def is_conversational_query(self, text: str) -> bool:
        """Determine if text is a conversational query vs desktop automation command using NLP."""
        return self.intent_classifier.classify_intent(text)


class VoiceHandler:
    """Handles speech recognition and text-to-speech functionality."""
    
    def __init__(self, config: Optional[AssistantConfig] = None):
        self.config = config or AssistantConfig()
        self.is_available = VOICE_AVAILABLE
        self.recognizer = None
        self.microphone = None
        self.tts_engine = None
        self.is_listening = False
        self.voice_queue = queue.Queue()
        self.listening_thread = None
        
        if VOICE_AVAILABLE:
            self._initialize_voice_components()
        else:
            logging.warning("Voice support not available - missing dependencies")
    
    def _initialize_voice_components(self):
        """Initialize speech recognition and TTS components."""
        try:
            # Initialize speech recognition
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            
            # Calibrate microphone for ambient noise
            with self.microphone as source:
                # Silent calibration for clean interface - longer duration for better calibration
                self.recognizer.adjust_for_ambient_noise(source, duration=2)
            
            # Adjust recognition sensitivity for better speech detection
            self.recognizer.energy_threshold = 300  # Lower threshold = more sensitive
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.dynamic_energy_adjustment_damping = 0.15
            self.recognizer.dynamic_energy_ratio = 1.5
            
            # Initialize text-to-speech
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', self.config.tts_rate)
            self.tts_engine.setProperty('volume', 1.0)  # Set volume to maximum
            
            # Set voice (try to use female voice if available)
            voices = self.tts_engine.getProperty('voices')
            if voices:
                # Try to find a female voice
                female_voice = next((voice for voice in voices if 'female' in voice.name.lower() or 'zira' in voice.name.lower()), None)
                if female_voice:
                    self.tts_engine.setProperty('voice', female_voice.id)
                else:
                    # Use first available voice
                    self.tts_engine.setProperty('voice', voices[0].id)
            
            # Voice components ready - silent initialization
            
        except Exception as e:
            logging.error(f"Failed to initialize voice components: {e}")
            self.is_available = False
    
    def speak(self, text: str) -> bool:
        """Convert text to speech."""
        if not self.is_available or not self.tts_engine:
            return False
        
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
            return True
        except Exception as e:
            return False
    
    def listen_once(self, timeout: Optional[float] = None) -> Optional[str]:
        """Listen for a single voice command."""
        if not self.is_available or not self.recognizer or not self.microphone:
            logging.warning("Speech recognition not available")
            return None
        
        timeout = timeout or self.config.voice_timeout
        
        try:
            print("Listening...")
            
            with self.microphone as source:
                # Listen for audio with longer timeouts for complete speech
                audio = self.recognizer.listen(source, 
                                             timeout=30,  # Wait up to 30 seconds for speech to start
                                             phrase_time_limit=20)  # Allow up to 20 seconds of continuous speech
            
            pass  # Silent processing
            
            # Recognize speech using Google's speech recognition
            text = self.recognizer.recognize_google(audio, language=self.config.voice_language)
            
            # Silent recognition for clean interface
            # Remove debug output for clean interface
            
            # Ensure microphone is fully released
            import time
            time.sleep(0.2)
            
            return text.strip()
            
        except sr.WaitTimeoutError:
            print("No speech detected (timeout)")
            return None
        except sr.UnknownValueError:
            print("Warning: Could not understand audio")
            return None
        except sr.RequestError as e:
            logging.error(f"Speech recognition error: {e}")
            print(f"Speech recognition error: {e}")
            return None
        except Exception as e:
            logging.error(f"Unexpected voice error: {e}")
            return None
    
    def start_continuous_listening(self, callback_func):
        """Start continuous listening in background thread."""
        if not self.is_available or self.is_listening:
            return False
        
        self.is_listening = True
        self.listening_thread = threading.Thread(
            target=self._continuous_listen_worker, 
            args=(callback_func,),
            daemon=True
        )
        self.listening_thread.start()
        print("Continuous listening started (say 'stop listening' to quit)")
        return True
    
    def stop_continuous_listening(self):
        """Stop continuous listening."""
        self.is_listening = False
        if self.listening_thread:
            self.listening_thread.join(timeout=2)
        print("Continuous listening stopped")
    
    def _continuous_listen_worker(self, callback_func):
        """Worker function for continuous listening."""
        while self.is_listening:
            try:
                text = self.listen_once(timeout=1.0)  # Short timeout for responsiveness
                
                if text:
                    # Check for stop command
                    if any(phrase in text.lower() for phrase in ['stop listening', 'quit voice', 'disable voice']):
                        print("Stop command detected")
                        self.is_listening = False
                        break
                    
                    # Send recognized text to callback
                    callback_func(text)
                
            except Exception as e:
                logging.error(f"Continuous listening error: {e}")
                continue
    
    def test_voice_setup(self) -> bool:
        """Test voice recognition and TTS setup."""
        if not self.is_available:
            print("Voice support not available")
            return False
        
        pass  # Silent voice setup
        
        # Test TTS
        if self.speak("Voice test. Can you hear me?"):
            pass  # Silent success
        else:
            print("Text-to-speech failed")
            return False
        
        # Test speech recognition
        pass  # Silent test
        text = self.listen_once(timeout=10)
        
        if text:
            pass  # Silent test success
            self.speak(f"I heard you say: {text}")
            return True
        else:
            print("Speech recognition failed or no input detected")
            return False
    
    def get_voice_info(self) -> Dict[str, Any]:
        """Get information about voice capabilities."""
        info = {
            "available": self.is_available,
            "listening": self.is_listening
        }
        
        if self.is_available and self.tts_engine:
            try:
                voices = self.tts_engine.getProperty('voices')
                info["tts_voices"] = len(voices) if voices else 0
                info["current_voice"] = self.tts_engine.getProperty('voice')
                info["tts_rate"] = self.tts_engine.getProperty('rate')
                info["tts_volume"] = self.tts_engine.getProperty('volume')
            except Exception:
                pass
        
        return info


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
        self.voice_handler = VoiceHandler(self.config) if self.config.voice_enabled else None
        
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
            logging.info("🔄 Detected multi-step task - will use iterative approach")
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
                logging.info(f"🔄 Multi-step iteration {iteration + 1}/{max_iterations}")
                
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
                    screen_w, screen_h = pyautogui.size()
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


# =============================================================================
# CLI Functions
# =============================================================================

def setup_logging(level: str = "ERROR"):
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
                print("API key saved to .env file")
            except Exception as e:
                print(f"Warning: Could not save to .env file: {e}")
        else:
            sys.exit(1)


def select_voice():
    """Let user select TTS voice or auto-select."""
    try:
        import pyttsx3
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        
        if not voices:
            print("No voices available, using default")
            return None
            
        print(f"Available voices:")
        for i, voice in enumerate(voices):
            # Get a clean voice name
            name = voice.name.split(' - ')[0] if ' - ' in voice.name else voice.name
            print(f"  {i}: {name}")
        
        # Find best default voice (prefer female or natural sounding)
        default_idx = 0
        for i, voice in enumerate(voices):
            voice_name = voice.name.lower()
            if any(keyword in voice_name for keyword in ['zira', 'hazel', 'aria', 'female']):
                default_idx = i
                break
        else:
            # If no preferred voice found, use second voice if available (often better than first)
            if len(voices) > 1:
                default_idx = 1
        
        default_voice = voices[default_idx]
        default_name = default_voice.name.split(' - ')[0] if ' - ' in default_voice.name else default_voice.name
        
        chosen = input(f"Choose voice (or press Enter for {default_name}): ").strip()
        
        if chosen == "":
            return default_voice.id
        
        try:
            voice_idx = int(chosen)
            if 0 <= voice_idx < len(voices):
                return voices[voice_idx].id
            else:
                print("Invalid voice number, using default")
                return default_voice.id
        except ValueError:
            print("Invalid input, using default")
            return default_voice.id
            
    except Exception as e:
        print(f"Voice selection error: {e}")
        return None


def interactive_mode():
    """Run the assistant in interactive mode."""
    load_environment()
    
    # Preload available models
    try:
        api_key = os.getenv('GOOGLE_AI_API_KEY')
        if api_key:
            with suppress_stderr():
                genai.configure(api_key=api_key)
                available = GeminiClient.list_available_models()
        else:
            available = []
    except Exception as e:
        available = []
        
    default_model = 'auto'
    if available:
        print(f"Available models: {', '.join(available)}")
        print(f"Recommended: {available[0] if available else 'gemini-1.5-flash'}")
    else:
        print("Available models: Will be auto-selected")
        
    chosen = input(f"Choose model (or press Enter for {default_model}): ").strip() or default_model
    
    # Voice selection
    selected_voice = select_voice()
    
    # Store selected voice globally for subprocess calls
    global SELECTED_VOICE_ID
    SELECTED_VOICE_ID = selected_voice
    
    config = AssistantConfig(
        safe_mode=False,
        confirmation_required=False,
        screenshot_dir=os.getenv('SCREENSHOT_DIR', 'screenshots'),
        model_name=chosen
    )
    
    # Initialize assistant and chatbot
    try:

        assistant = DesktopAssistant(config)
        chatbot = ChatBot(assistant.gemini_client)
        pass  # Clean startup - no system messages
    except Exception as e:
        print(f"Error: Failed to initialize Truvo: {e}")
        return
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
                
            # Handle commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
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
            elif user_input.lower() in ['chat-history', 'history']:
                print(f"Chat history: {chatbot.get_history_summary()}")
                continue
            elif user_input.lower() in ['clear-chat', 'clear-history']:
                chatbot.clear_history()
                print("Conversation history cleared!")
                continue
            elif user_input.lower().startswith('config'):
                handle_config_command(assistant, user_input)
                continue
            # Voice commands
            elif user_input.lower() in ['voice', 'enable voice', 'voice on']:
                toggle_voice_mode(assistant, True)
                continue
            elif user_input.lower() in ['no voice', 'disable voice', 'voice off']:
                toggle_voice_mode(assistant, False)
                continue
            elif user_input.lower() in ['voice test', 'test voice']:
                test_voice_setup(assistant)
                continue
            elif user_input.lower() in ['voice listen', 'listen', 'voice input']:
                handle_voice_input(assistant, chatbot)
                continue
            elif user_input.lower() in ['voice mode', 'continuous voice', 'always listen']:
                start_continuous_voice_mode(assistant, chatbot)
                continue
            elif user_input.lower() in ['voice chat', 'interactive voice', 'voice conversation']:
                start_interactive_voice_mode(assistant, chatbot)
                continue
            
            # Determine if this is a conversation or automation task using NLP
            is_chat = chatbot.is_conversational_query(user_input)
            
            if is_chat:
                # Handle as conversation
                response = chatbot.chat(user_input)
                print(f"Truvo: {response}")
                
                # Voice response if voice is enabled (silent)
                if assistant.voice_handler and assistant.voice_handler.is_available:
                    assistant.voice_handler.speak(response)
            else:
                # Handle as automation task
                result = assistant.execute_task(user_input)
                
                # Show results
                print("\n" + "="*60)
                if result["success"]:
                    print(f"SUCCESS - Task completed!")
                    print(f"   Actions: {result['actions_completed']}/{result['total_actions']}")
                else:
                    print(f"FAILED - {result.get('error', 'Unknown error')}")
                    print(f"   Actions completed: {result['actions_completed']}")
                    
                    if result.get("safety_concerns"):
                        print(f"Safety concerns: {', '.join(result['safety_concerns'])}")
                print("="*60)
            
        except KeyboardInterrupt:
            print("\n\nTask interrupted by user")
            assistant.stop_task()
        except EOFError:
            print("Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


def show_help():
    """Show help information."""
    print("""
╔══════════════════════════════════════════════════════════╗
║                         HELP                             ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║ TASK EXAMPLES:                                           ║
║   • "Take a screenshot"                                  ║
║   • "Open notepad and type hello world"                 ║
║   • "Click on the Chrome icon"                          ║
║   • "Press the Windows key"                             ║
║   • "Scroll down 3 times"                               ║
║                                                          ║
║ CONTROL COMMANDS:                                        ║
║   • help         - Show this help                       ║
║   • status       - Show assistant status                ║
║   • analyze      - Analyze current screen               ║
║   • chat-history - Show conversation history            ║
║   • clear-chat   - Clear conversation history           ║
║   • quit         - Exit Truvo                           ║
║                                                          ║
║ VOICE COMMANDS:                                          ║
║   • voice        - Enable voice mode                    ║
║   • voice off    - Disable voice mode                   ║
║   • voice test   - Test voice setup                     ║
║   • listen       - Listen for voice input once          ║
║   • voice chat   - Interactive voice conversation       ║
║   • always listen- Continuous voice mode                ║
║   • voice status - Show voice capabilities              ║
║                                                          ║
║ CONVERSATION:                                            ║
║   • Ask me anything! I can chat about any topic         ║
║   • "What's the weather like?"                          ║
║   • "Tell me a joke"                                    ║
║   • "How do neural networks work?"                      ║
║   • "What do you think about..."                        ║
║                                                          ║
║ CONFIGURATION:                                           ║
║   • config safe on/off      - Toggle safe mode         ║
║   • config confirm on/off   - Toggle confirmations     ║
║   • config voice on/off     - Toggle voice mode        ║
║   • config show             - Show current config      ║
║                                                          ║
║ TIPS:                                                    ║
║   • Be specific about what you want                     ║
║   • Use Ctrl+C to stop running tasks                    ║
║   • Screenshots are saved automatically                 ║
║   • Voice responses can be enabled/disabled             ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""")


def show_status(assistant: DesktopAssistant):
    """Show assistant status."""
    status = assistant.get_task_status()
    config = assistant.config
    
    print(f"""
╔══════════════════════════════════════════════════════════╗
║                      TRUVO STATUS                        ║
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
    print("Analyzing current screen...")
    
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
            print(f"\nSafety Concerns: {', '.join(analysis['safety_concerns'])}")
            
        print(f"\nConfidence: {analysis.get('overall_confidence', 0.0):.2f}")
        print("═"*60)
        
    except Exception as e:
        print(f"Error analyzing screen: {e}")


def toggle_voice_mode(assistant: DesktopAssistant, enable: bool):
    """Toggle voice mode on/off."""
    assistant.config.voice_enabled = enable
    
    if enable:
        if not VOICE_AVAILABLE:
            print("Voice libraries not installed. Install with: pip install SpeechRecognition pyttsx3 pyaudio")
            return
        
        if not assistant.voice_handler:
            assistant.voice_handler = VoiceHandler(assistant.config)
        
        if assistant.voice_handler.is_available:
            assistant.voice_handler.speak("Voice mode enabled.")
        else:
            print("Voice initialization failed")
    else:
        print("Voice mode disabled")
        assistant.voice_handler = None


def test_voice_setup(assistant: DesktopAssistant):
    """Test voice recognition and TTS setup."""
    if not VOICE_AVAILABLE:
        print("Voice libraries not installed")
        return
    
    if not assistant.voice_handler:
        print("Initializing voice handler for testing...")
        temp_handler = VoiceHandler(assistant.config)
    else:
        temp_handler = assistant.voice_handler
    
    success = temp_handler.test_voice_setup()
    
    if success:
        pass  # Silent success
    else:
        print("Voice setup test failed. Check your microphone and speakers.")


def start_continuous_voice_mode(assistant: DesktopAssistant, chatbot: ChatBot):
    """Start continuous voice listening mode."""
    # Auto-enable voice if not available
    if not assistant.voice_handler or not assistant.voice_handler.is_available:
        print("Enabling voice mode for continuous listening...")
        toggle_voice_mode(assistant, True)
        if not assistant.voice_handler or not assistant.voice_handler.is_available:
            print("Failed to enable voice mode. Check your microphone and speakers.")
            return
    
    pass  # Silent activation
    
    assistant.voice_handler.speak("Continuous voice mode activated. I'm listening for your commands.")
    
    def voice_callback(text):
        """Process voice input in continuous mode."""
        print(f"You: {text}")
        
        # Check for exit commands
        exit_phrases = ['stop listening', 'quit voice', 'exit voice', 'disable voice', 'voice off']
        if any(phrase in text.lower() for phrase in exit_phrases):
            assistant.voice_handler.stop_continuous_listening()
            assistant.voice_handler.speak("Voice mode disabled. Returning to text mode.")
            return
        
        # Process the voice input
        is_chat = chatbot.is_conversational_query(text)
        
        if is_chat:
            print("Processing...")
            response = chatbot.chat(text)
            print(f"Truvo: {response}")
            # Voice after text is shown
            assistant.voice_handler.speak(response)
        else:
            print("Executing...")
            assistant.voice_handler.speak("Executing your request.")
            
            result = assistant.execute_task(text)
            
            if result["success"]:
                success_msg = f"Task completed. {result['actions_completed']} actions performed."
                print(f"{success_msg}")
                assistant.voice_handler.speak(success_msg)
            else:
                error_msg = f"Task failed: {result.get('error', 'Unknown error')}"
                print(f"Error: {error_msg}")
                assistant.voice_handler.speak(error_msg)
        
        print("─" * 30)
    
    try:
        # Start continuous listening
        assistant.voice_handler.start_continuous_listening(voice_callback)
        
        # Keep the main thread alive
        import time
        while assistant.voice_handler.is_listening:
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        assistant.voice_handler.stop_continuous_listening()
    
    pass  # Silent exit


def start_interactive_voice_mode(assistant: DesktopAssistant, chatbot: ChatBot):
    """Start interactive voice chat mode with prompts."""
    # Auto-enable voice if not available
    if not assistant.voice_handler or not assistant.voice_handler.is_available:
        pass  # Silent enable
        toggle_voice_mode(assistant, True)
        if not assistant.voice_handler or not assistant.voice_handler.is_available:
            print("Failed to enable voice mode. Check your microphone and speakers.")
            return
    
    # Interactive voice mode - minimal UI
    # Start directly without initialization messages
    
    try:
        while True:
            # Prompt for voice input - silent
            
            # Listen for input
            text = assistant.voice_handler.listen_once(timeout=15)
            
            if not text:
                assistant.voice_handler.speak("I didn't hear anything. Let's try again.")
                continue
            
            print(f"You said: {text}")
            
            # Check for exit
            if any(word in text.lower() for word in ['quit', 'exit', 'stop', 'goodbye', 'bye']):
                assistant.voice_handler.speak("Goodbye!")
                break
            
            # Process input
            is_chat = chatbot.is_conversational_query(text)
            
            if is_chat:
                response = chatbot.chat(text)
                print(f"Truvo: {response}")
                # Voice response - use external TTS process to avoid conflicts
                if assistant.voice_handler and assistant.voice_handler.is_available:
                    try:
                        import subprocess
                        # Use complete response for voice - no truncation
                        voice_response = response.replace('\n', ' ').replace('  ', ' ').strip()
                        
                        # Use external TTS script with no timeout - let it complete fully
                        cmd = [sys.executable, "speak.py", voice_response]
                        if SELECTED_VOICE_ID:
                            cmd.append(SELECTED_VOICE_ID)
                        subprocess.run(cmd, 
                                     cwd="c:\\Users\\deb0p\\truvo",
                                     # No timeout - let it speak completely
                                     capture_output=True)
                    except Exception as e:
                        print(f"External TTS Error: {e}")
            else:
                print("Executing task...")
                assistant.voice_handler.speak("I'll execute that task for you.")
                
                result = assistant.execute_task(text)
                
                if result["success"]:
                    success_msg = f"Done! {result['actions_completed']} actions completed."
                    print(f"{success_msg}")
                    assistant.voice_handler.speak(success_msg)
                else:
                    error_msg = f"Sorry, that didn't work: {result.get('error', 'Unknown error')}"
                    print(f"Error: {error_msg}")
                    assistant.voice_handler.speak(error_msg)
            
            # Clean interface - no separators
            
    except KeyboardInterrupt:
        assistant.voice_handler.speak("Goodbye!")
    
    pass  # Silent exit


def handle_voice_input(assistant: DesktopAssistant, chatbot: ChatBot):
    """Handle a single voice input and process it."""
    if not assistant.voice_handler or not assistant.voice_handler.is_available:
        print("Voice mode not available. Type 'voice' to enable it.")
        return
    
    # Listen for voice input - clean interface
    text = assistant.voice_handler.listen_once()
    
    if text:
        print(f"Voice input: {text}")
        
        # Process the voice input like regular text input
        is_chat = chatbot.is_conversational_query(text)
        
        if is_chat:
            # Handle as conversation
            print(f"\nProcessing conversation...")
            response = chatbot.chat(text)
            print(f"Truvo: {response}")
            assistant.voice_handler.speak(response)
        else:
            # Handle as automation task
            print(f"\nExecuting automation task...")
            assistant.voice_handler.speak("Executing your request.")
            
            result = assistant.execute_task(text)
            
            if result["success"]:
                success_msg = f"Task completed successfully. {result['actions_completed']} actions performed."
                print(f"{success_msg}")
                assistant.voice_handler.speak(success_msg)
            else:
                error_msg = f"Task failed: {result.get('error', 'Unknown error')}"
                print(f"Error: {error_msg}")
                assistant.voice_handler.speak(error_msg)
    else:
        print("No voice input detected")
    
    print("─" * 40)
    print()


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
            print("Safe mode enabled")
        elif value in ['off', 'false', 'no']:
            assistant.set_safe_mode(False)
            print("Safe mode disabled")
        else:
            print("Usage: config safe on/off")
    
    elif setting in ['confirm', 'confirmation']:
        if value in ['on', 'true', 'yes']:
            assistant.set_confirmation_required(True)
            print("Action confirmation enabled")
        elif value in ['off', 'false', 'no']:
            assistant.set_confirmation_required(False)
            print("Action confirmation disabled")
        else:
            print("Usage: config confirm on/off")
    
    elif setting == 'voice':
        if value in ['on', 'true', 'yes', 'enable']:
            toggle_voice_mode(assistant, True)
        elif value in ['off', 'false', 'no', 'disable']:
            toggle_voice_mode(assistant, False)
        elif value == 'status':
            if assistant.voice_handler:
                info = assistant.voice_handler.get_voice_info()
                print(f"Voice Status: {'Available' if info['available'] else 'Not Available'}")
                if info['available']:
                    print(f"   Listening: {'Yes' if info['listening'] else 'No'}")
                    print(f"   TTS Voices: {info.get('tts_voices', 'Unknown')}")
                    print(f"   TTS Rate: {info.get('tts_rate', 'Unknown')} WPM")
            else:
                print("Voice mode disabled")
        else:
            print("Usage: config voice on/off/status")
    
    else:
        print(f"Unknown setting: {setting}")


def single_task_mode(task: str):
    """Execute a single task and exit."""
    print(f"Executing single task: {task}")
    model = os.getenv('ASSISTANT_MODEL', 'auto')
    config = AssistantConfig(
        safe_mode=False,
        confirmation_required=False,
        screenshot_dir=os.getenv('SCREENSHOT_DIR', 'screenshots'),
        model_name=model
    )
    
    try:
        assistant = DesktopAssistant(config)
        result = assistant.execute_task(task)
        
        if result["success"]:
            print(f"Task completed successfully!")
            print(f"Actions: {result['actions_completed']}/{result['total_actions']}")
        else:
            print(f"Task failed: {result.get('error', 'Unknown error')}")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Truvo - AI-powered desktop automation assistant",
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
        default='ERROR',
        help='Set logging level'
    )
    
    parser.add_argument(
        '--no-safe-mode',
        action='store_true',
        help='Disable safe mode'
    )
    parser.add_argument(
        '--model',
        default='auto',
        help='Specify Gemini model name or auto for best available'
    )
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List available models and exit'
    )
    
    args = parser.parse_args()
    
    # Setup
    setup_logging(args.log_level)
    load_environment()
    
    if args.no_safe_mode:
        os.environ['SAFE_MODE'] = 'false'

    # Handle model listing
    if args.list_models:
        try:
            models = GeminiClient.list_available_models()
            if models:
                print("Available models:")
                for m in models:
                    sel = '*' if m == args.model else ' '
                    print(f" {sel} {m}")
            else:
                print("No models retrieved (check API key / network)")
        except Exception as e:
            print(f"Failed to list models: {e}")
        return

    # Persist chosen model to env for single_task_mode helper
    os.environ['ASSISTANT_MODEL'] = args.model
    
    # Run appropriate mode
    try:
        if args.analyze:
            config = AssistantConfig(model_name=args.model)
            assistant = DesktopAssistant(config)
            analyze_screen(assistant)
        elif args.task:
            single_task_mode(args.task)
        else:
            interactive_mode()
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()