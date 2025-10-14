#!/usr/bin/env python3
"""
Voice Handler Module for Desktop Assistant
Handles speech recognition and text-to-speech functionality.
"""
import sys
import logging
import threading
import queue
from typing import Optional, Dict, Any
from dataclasses import dataclass

# Voice recognition imports
try:
    import speech_recognition as sr
    import pyttsx3
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False
    print("Warning: Voice libraries not installed. Voice mode disabled. Install with: pip install SpeechRecognition pyttsx3 pyaudio")

# Try to import AssistantConfig from main.py, otherwise create a lightweight version
try:
    from main import AssistantConfig
except ImportError:
    # Fallback: Create lightweight config if main.py not available
    @dataclass
    class AssistantConfig:
        """Configuration for voice handler (lightweight version for interfaces/speak.py)."""
        voice_enabled: bool = True
        voice_language: str = 'en-US'
        voice_timeout: float = 5.0
        tts_rate: int = 150


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


# Standalone TTS script functionality (for backward compatibility)
if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)
    
    text = sys.argv[1]
    voice_id = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        engine = pyttsx3.init()
        
        # Set specific voice if provided, otherwise auto-select
        if voice_id:
            engine.setProperty('voice', voice_id)
        else:
            # Get available voices and try to find a better one
            voices = engine.getProperty('voices')
            if voices:
                # Try to find a female voice or more natural sounding voice
                for voice in voices:
                    voice_name = voice.name.lower()
                    # Look for names that indicate better voices
                    if any(keyword in voice_name for keyword in ['zira', 'hazel', 'aria', 'female', 'natural']):
                        engine.setProperty('voice', voice.id)
                        break
                else:
                    # If no preferred voice found, use the second voice if available (often better than first)
                    if len(voices) > 1:
                        engine.setProperty('voice', voices[1].id)
        
        # Set natural speech properties - optimized for long responses
        engine.setProperty('rate', 170)  # Comfortable rate for long text
        engine.setProperty('volume', 0.95)
        
        # Handle any length of text - no restrictions
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        # Fail silently for voice issues
        sys.exit(1)