#!/usr/bin/env python3
"""
Standalone TTS script for Truvo Desktop Assistant
Handles text-to-speech functionality as an external process.
Supports multiple languages using Google Text-to-Speech.
"""
import sys
import os
import tempfile
import subprocess

# Try to import audio playback libraries
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

# Try to import gTTS for multilingual support
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

# Try to import language detection
try:
    from llm.gemini.translation import detect_language
    LANG_DETECT_AVAILABLE = True
except ImportError:
    LANG_DETECT_AVAILABLE = False

# Language mapping for gTTS
LANGUAGE_MAP = {
    'english': 'en',
    'hindi': 'hi',
    'spanish': 'es',
    'french': 'fr',
    'german': 'de',
    'italian': 'it',
    'portuguese': 'pt',
    'russian': 'ru',
    'japanese': 'ja',
    'korean': 'ko',
    'chinese': 'zh-cn',
    'arabic': 'ar',
    'bengali': 'bn',
    'gujarati': 'gu',
    'kannada': 'kn',
    'malayalam': 'ml',
    'marathi': 'mr',
    'nepali': 'ne',
    'punjabi': 'pa',
    'sanskrit': 'sa',
    'tamil': 'ta',
    'telugu': 'te',
    'urdu': 'ur'
}

def detect_text_language(text: str) -> str:
    """Detect the language of the input text."""
    if LANG_DETECT_AVAILABLE:
        try:
            detected = detect_language(text)
            return detected.lower()
        except Exception:
            pass

    # Fallback: simple heuristic based on script
    if any('\u0900' <= char <= '\u097F' for char in text):  # Devanagari script (Hindi)
        return 'hindi'
    elif any('\u0980' <= char <= '\u09FF' for char in text):  # Bengali
        return 'bengali'
    elif any('\u0A80' <= char <= '\u0AFF' for char in text):  # Gujarati
        return 'gujarati'
    elif any('\u0C80' <= char <= '\u0CFF' for char in text):  # Kannada
        return 'kannada'
    elif any('\u0D00' <= char <= '\u0D7F' for char in text):  # Malayalam
        return 'malayalam'
    elif any('\u0B80' <= char <= '\u0BFF' for char in text):  # Tamil
        return 'tamil'
    elif any('\u0C00' <= char <= '\u0C7F' for char in text):  # Telugu
        return 'telugu'
    else:
        return 'english'  # Default to English

def get_gtts_lang(language: str) -> str:
    """Map detected language to gTTS language code."""
    return LANGUAGE_MAP.get(language.lower(), 'en')

def play_audio(file_path: str):
    """Play audio file using pygame (background playback, no windows)."""
    if PYGAME_AVAILABLE:
        try:
            pygame.mixer.init()
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()
            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                pygame.time.wait(100)
            pygame.mixer.quit()
            return True
        except Exception as e:
            print(f"pygame playback failed: {e}")

    # Fallback: try Windows Media Player (will open window)
    try:
        subprocess.run(['cmd', '/c', 'start', '/wait', file_path], check=True)
        return True
    except Exception:
        # Last resort: try PowerShell method
        try:
            ps_command = f'(New-Object Media.SoundPlayer "{file_path}").PlaySync()'
            subprocess.run(['powershell', '-Command', ps_command], check=True)
            return True
        except Exception:
            # Final fallback: just open the file
            os.startfile(file_path)
            return False

def speak_with_gtts(text: str, lang: str = None):
    """Speak text using Google Text-to-Speech."""
    if not GTTS_AVAILABLE:
        print("gTTS not available, falling back to pyttsx3")
        return speak_with_pyttsx3(text)

    try:
        # Detect language if not provided
        if not lang:
            detected_lang = detect_text_language(text)
            lang = get_gtts_lang(detected_lang)

        # Create gTTS object
        tts = gTTS(text=text, lang=lang, slow=False)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
            temp_path = temp_file.name
            tts.save(temp_path)

        # Play the audio
        play_audio(temp_path)

        # Clean up
        try:
            os.unlink(temp_path)
        except:
            pass

    except Exception as e:
        print(f"gTTS failed: {e}, falling back to pyttsx3")
        speak_with_pyttsx3(text)

def speak_with_pyttsx3(text: str):
    """Fallback TTS using pyttsx3 (English only)."""
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty('rate', 170)
        engine.setProperty('volume', 0.95)

        # Try to select a good voice
        voices = engine.getProperty('voices')
        if voices:
            for voice in voices:
                voice_name = voice.name.lower()
                if any(keyword in voice_name for keyword in ['zira', 'hazel', 'aria', 'female', 'natural']):
                    engine.setProperty('voice', voice.id)
                    break
            else:
                if len(voices) > 1:
                    engine.setProperty('voice', voices[1].id)

        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        print(f"pyttsx3 TTS failed: {e}")

def main():
    if len(sys.argv) < 2:
        sys.exit(1)

    text = sys.argv[1]
    voice_id = sys.argv[2] if len(sys.argv) > 2 else None

    # Use gTTS for multilingual support, fallback to pyttsx3
    if GTTS_AVAILABLE:
        speak_with_gtts(text)
    else:
        speak_with_pyttsx3(text)

if __name__ == "__main__":
    main()