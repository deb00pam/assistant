"""
Configuration module for Desktop Assistant

This module contains the AssistantConfig dataclass that defines
all configuration settings for the desktop assistant, and handles
environment variable setup.
"""

import os
from dataclasses import dataclass


_environment_setup_done = False

def setup_environment():
    """
    Set up all required environment variables for the application.
    
    This function should be called at the very start of the application
    to configure environment variables for:
    - Google AI library warning suppression
    - LangChain USER_AGENT
    - Other runtime configurations
    
    Call this before importing any other modules that depend on these settings.
    This function is idempotent - it can be called multiple times safely.
    """
    global _environment_setup_done
    
    # Only run once to avoid redundant setup
    if _environment_setup_done:
        return
    
    # Suppress Google AI library warnings
    os.environ['GRPC_VERBOSITY'] = 'ERROR'
    os.environ['GLOG_minloglevel'] = '3'  # More aggressive suppression
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['GRPC_TRACE'] = ''
    os.environ['GRPC_VERBOSITY'] = 'NONE'
    
    # Set USER_AGENT for LangChain web retrieval
    if 'USER_AGENT' not in os.environ:
        os.environ['USER_AGENT'] = 'Desktop-Assistant/1.0 (https://github.com/deb00pam/truvo)'
    
    _environment_setup_done = True


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

    # Multilingual support
    user_language: str = "auto"  # User's preferred language (e.g., 'en', 'es', 'fr', 'auto' for detect)
    assistant_language: str = "en"  # Assistant's response language (default English)
