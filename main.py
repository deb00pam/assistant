#!/usr/bin/env python3
"""
Truvo - AI-Powered Desktop Assistant

A self-operating desktop assistant powered by Google's Gemini AI that can see,
understand, and interact with your desktop environment through natural language commands.
"""

# Set up environment variables FIRST before any other imports
from core.config import setup_environment
setup_environment()

# Now import standard libraries
import os
import warnings
import logging as default_logging
import re
import json
from typing import Dict, List, Any, Optional, Tuple

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
from automation.storage import load_user_data, save_user_data

# Load user data at startup
USER_DATA = load_user_data()

# Import translation utilities - now powered by Gemini AI!
try:
    from llm.gemini.translation import detect_language, translate_text
    TRANSLATION_AVAILABLE = True
except Exception as e:
    TRANSLATION_AVAILABLE = False
    print(f"Translation not available: {e}")
    # Create fallback functions
# Translation now handled by llm.gemini.translation module

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

# Note: Traditional IntentClassifier (PostgreSQL-based) is no longer used
# We now use GeminiIntentClassifier for AI-powered intent classification

# Import GeminiIntentClassifier for AI-powered intent classification
try:
    from llm.gemini.intent_classifier import GeminiIntentClassifier
except ImportError:
    GeminiIntentClassifier = None

# Import GeminiClient from llm package
try:
    from llm.gemini.client import GeminiClient
except ImportError:
    GeminiClient = None

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

# Import VoiceHandler from speak.py
try:
    from interfaces.voice import VoiceHandler
except ImportError:
    VoiceHandler = None

# OS Detection import
try:
    from automation.os_detection import get_os_context, get_os_commands, os_detector
    OS_DETECTION_AVAILABLE = True
except ImportError:
    OS_DETECTION_AVAILABLE = False
    print("Warning: OS detection not available")

# Data Retrieval import - Gemini AI powered local data retrieval
try:
    from llm.gemini.local_retrieval import GeminiLocalDataRetriever
    LOCAL_DATA_AVAILABLE = True
except ImportError as e:
    LOCAL_DATA_AVAILABLE = False
    print(f"Gemini local data retrieval not available: {e}")
    GeminiLocalDataRetriever = None

# Web Data Retrieval - Now using Gemini AI (llm/gemini_web_retrieval.py)
# Legacy data.web_retrieval module removed - using Gemini AI instead
WEB_DATA_AVAILABLE = True  # Gemini web retrieval is always available

# Combined availability
DATA_RETRIEVAL_AVAILABLE = LOCAL_DATA_AVAILABLE or WEB_DATA_AVAILABLE


# =============================================================================
# Configuration and Data Classes
# =============================================================================

# Import configuration and core components
try:
    from core.config import AssistantConfig
except ImportError:
    AssistantConfig = None

# Import ChatBot from chatbot.py
try:
    from core.chatbot import ChatBot
except ImportError:
    ChatBot = None

# Import DesktopAssistant from core.desktop_assistant
try:
    from core.desktop_assistant import DesktopAssistant
except ImportError:
    DesktopAssistant = None


# ChatBot class has been moved to chatbot.py
# DesktopAssistant class has been moved to desktop_assistant.py


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
        # Recommend the best model with quota info
        recommended = available[0] if available else 'gemini-1.5-flash'
        if 'gemini-1.5-flash' in available:
            recommended = 'gemini-1.5-flash'
            print(f"Recommended: {recommended} (1500 requests/day quota)")
        elif 'gemini-1.5-flash-8b' in available:
            recommended = 'gemini-1.5-flash-8b' 
            print(f"Recommended: {recommended} (good quota limits)")
        else:
            print(f"Recommended: {recommended}")
    else:
        print("Available models: Will be auto-selected with quota awareness")
        
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
        
        # Initialize Gemini Intent Classifier using the same model as assistant
        if GeminiIntentClassifier:
            gemini_intent_classifier = GeminiIntentClassifier(gemini_client=assistant.gemini_client)
        else:
            gemini_intent_classifier = None
            print("Gemini Intent Classifier not available, using fallback")
            
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
            
            # NEW: Use 4-way AI-powered intent classification
            if gemini_intent_classifier:
                try:
                    intent = gemini_intent_classifier.classify_intent(user_input)
                except Exception as e:
                    # Fallback to conversation on classification error
                    intent = "conversation"
            else:
                # Use GeminiIntentClassifier for proper intent detection
                try:
                    intent_result = intent_classifier.classify_with_confidence(user_input)
                    intent = intent_result.get('intent', 'conversation')
                except Exception:
                    intent = "conversation"  # Safe fallback
            
            # Handle based on intent type
            if intent == "conversation":
                # Handle as conversation
                response = chatbot.chat(user_input)
                print(f"Truvo: {response}")
                
                # Voice response if voice is enabled (silent)
                if assistant.voice_handler and assistant.voice_handler.is_available:
                    assistant.voice_handler.speak(response)
                    
            elif intent == "automation":
                # Handle as automation task
                result = assistant.execute_task(user_input)
                
                # Show results
                print("\n" + "="*60)
                if result["success"]:
                    print(f"SUCCESS - Task completed!")
                    print(f"   Actions: {result['actions_completed']}/{result['total_actions']}")
                else:
                    print(f"FAILED - {result['error']}")
                    if result['actions_completed'] > 0:
                        print(f"   Partial completion: {result['actions_completed']}/{result['total_actions']}")
                print("="*60)
                
            elif intent == "local_data_retrieval":
                # Handle local data search using Gemini AI
                try:
                    if LOCAL_DATA_AVAILABLE:
                        # Create Gemini local data retriever (share the same model as assistant)
                        local_retriever = GeminiLocalDataRetriever(gemini_client=assistant.gemini_client)
                        response = local_retriever.retrieve_local_data(user_input)
                        print(f"Local Data Result: {response}")
                    else:
                        print("Gemini local data retrieval not available. Check dependencies.")
                except Exception as e:
                    print(f"Gemini local data retrieval failed: {e}")
                    # Fallback to conversation
                    response = chatbot.chat(user_input)
                    print(f"Truvo: {response}")
                    
            elif intent == "web_data_retrieval":
                # Handle web search using Gemini AI
                try:
                    from llm.gemini.web_retrieval import web_search
                    response = web_search(user_input, gemini_client=assistant.gemini_client)
                    print(f"Web Search Result: {response}")
                except Exception as e:
                    print(f"Web search failed: {e}")
                    # Fallback to conversation
                    response = chatbot.chat(user_input)
                    print(f"Truvo: {response}")
                    
            else:
                # Fallback - treat as conversation
                response = chatbot.chat(user_input)
                print(f"Truvo: {response}")
            
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
        
        # Process the voice input using proper intent classification
        try:
            intent_result = intent_classifier.classify_with_confidence(text)
            intent = intent_result.get('intent', 'conversation')
            is_chat = intent == 'conversation'
        except Exception:
            is_chat = True  # Safe fallback to conversation
        
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
            # Classify the input properly
            try:
                intent_result = intent_classifier.classify_with_confidence(text)
                intent = intent_result.get('intent', 'conversation')
                is_chat = intent == 'conversation'
            except Exception:
                is_chat = True  # Safe fallback to conversation
            
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
        # Classify the input properly  
        try:
            intent_result = intent_classifier.classify_with_confidence(text)
            intent = intent_result.get('intent', 'conversation')
            is_chat = intent == 'conversation'
        except Exception:
            is_chat = True  # Safe fallback to conversation
        
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


def multiple_tasks_mode(tasks: List[str]):
    """Execute multiple tasks sequentially."""
    print(f"Executing {len(tasks)} tasks sequentially:")
    for i, task in enumerate(tasks, 1):
        print(f"\n--- Task {i}/{len(tasks)}: {task} ---")
    
    model = os.getenv('ASSISTANT_MODEL', 'auto')
    config = AssistantConfig(
        safe_mode=False,
        confirmation_required=False,
        screenshot_dir=os.getenv('SCREENSHOT_DIR', 'screenshots'),
        model_name=model
    )
    
    try:
        assistant = DesktopAssistant(config)
        total_completed = 0
        total_actions = 0
        
        for i, task in enumerate(tasks, 1):
            print(f"\nExecuting task {i}/{len(tasks)}: {task}")
            result = assistant.execute_task(task)
            
            if result["success"]:
                print(f"✓ Task {i} completed successfully!")
                print(f"  Actions: {result['actions_completed']}/{result['total_actions']}")
                total_completed += 1
                total_actions += result['total_actions']
            else:
                print(f"✗ Task {i} failed: {result.get('error', 'Unknown error')}")
                if result['actions_completed'] > 0:
                    print(f"  Partial completion: {result['actions_completed']}/{result['total_actions']}")
                    total_actions += result['total_actions']
                # Continue with next task even if one fails
        
        print(f"\n{'='*60}")
        print(f"Multiple tasks execution complete!")
        print(f"Tasks completed: {total_completed}/{len(tasks)}")
        print(f"Total actions executed: {total_actions}")
        print(f"{'='*60}")
        
        if total_completed == len(tasks):
            print("All tasks completed successfully!")
        else:
            print(f"{len(tasks) - total_completed} task(s) failed or had issues.")
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
  python main.py --tasks "open notepad;type hello;save file as test.txt"  # Multiple tasks
  python main.py --analyze                   # Analyze screen only
        """
    )
    
    parser.add_argument(
        '--task', '-t',
        help='Execute a single task and exit'
    )
    
    parser.add_argument(
        '--tasks',
        help='Execute multiple tasks sequentially (semicolon-separated)'
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
        elif args.tasks:
            # Parse multiple tasks separated by semicolons
            tasks = [task.strip() for task in args.tasks.split(';') if task.strip()]
            if not tasks:
                print("Error: No valid tasks provided")
                sys.exit(1)
            multiple_tasks_mode(tasks)
        else:
            interactive_mode()
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()