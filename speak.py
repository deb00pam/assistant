#!/usr/bin/env python3
"""
Standalone TTS script for Truvo voice responses
"""
import sys
import pyttsx3

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