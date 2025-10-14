
#!/usr/bin/env python3
"""
Gemini AI Translation Module for Desktop Assistant
Uses Google's Gemini AI for language detection and translation.
"""

import os
import sys
from typing import Optional

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Import Gemini AI infrastructure  
try:
    # Add the parent directory to Python path to ensure imports work
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from .client import GeminiClient
    from core.config import AssistantConfig
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError as e:
    print(f"Gemini AI not available for translation: {e}")
    GEMINI_AVAILABLE = False

class GeminiTranslator:
    """Gemini AI-powered translation and language detection"""
    
    def __init__(self, gemini_client=None):
        """
        Initialize the Gemini Translator
        
        Args:
            gemini_client: Existing GeminiClient instance to reuse (optional)
        """
        if not GEMINI_AVAILABLE:
            raise ImportError("Gemini AI not available. Check dependencies.")
        
        try:
            if gemini_client is not None:
                # Use the provided GeminiClient instance (shares same model)
                self.gemini_client = gemini_client
                self.model = gemini_client.model
            else:
                # Create our own GeminiClient instance
                config = AssistantConfig(model_name="auto")
                self.gemini_client = GeminiClient(config=config)
                self.model = self.gemini_client.model
        except Exception as e:
            print(f"Error initializing Gemini Translator: {e}")
            raise
    
    def detect_language(self, text: str) -> str:
        """
        Detect the language of input text using Gemini AI
        
        Args:
            text: Text to analyze
            
        Returns:
            Language code (e.g., 'en', 'es', 'fr', 'de', etc.)
        """
        if not text or not text.strip():
            return "en"  # Default to English for empty text
        
        prompt = f"""Detect the language of this text and respond with ONLY the 2-letter ISO language code.

Text to analyze: "{text}"

Language code:"""

        try:
            response = self.model.generate_content(prompt)
            detected_lang = response.text.strip().lower()
            
            # Validate the response is a reasonable language code
            if len(detected_lang) == 2 and detected_lang.isalpha():
                return detected_lang
            else:
                # Fallback parsing for common languages
                response_lower = response.text.lower()
                if "english" in response_lower:
                    return "en"
                elif "spanish" in response_lower:
                    return "es"
                elif "french" in response_lower:
                    return "fr"
                elif "german" in response_lower:
                    return "de"
                else:
                    print(f"Unclear language detection: '{response.text}', defaulting to English")
                    return "en"
                    
        except Exception as e:
            print(f"Language detection error: {e}")
            return "en"  # Default to English on error
    
    def translate_text(self, text: str, target_lang: str, source_lang: str = None) -> str:
        """
        Translate text using Gemini AI
        
        Args:
            text: Text to translate
            target_lang: Target language code (e.g., 'en', 'es', 'fr')
            source_lang: Source language code (optional, auto-detect if None)
            
        Returns:
            Translated text
        """
        if not text or not text.strip():
            return text
        
        # Language name mapping for better prompts
        lang_names = {
            'en': 'English',
            'es': 'Spanish', 
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'zh': 'Chinese',
            'ja': 'Japanese',
            'ko': 'Korean',
            'ar': 'Arabic',
            'hi': 'Hindi'
        }
        
        target_name = lang_names.get(target_lang, target_lang.upper())
        
        if source_lang:
            source_name = lang_names.get(source_lang, source_lang.upper())
            prompt = f"""Translate the following {source_name} text to {target_name}.
Provide ONLY the translation, no explanations or additional text.

Text to translate: "{text}"

Translation:"""
        else:
            prompt = f"""Translate the following text to {target_name}.
Provide ONLY the translation, no explanations or additional text.

Text to translate: "{text}"

Translation:"""

        try:
            response = self.model.generate_content(prompt)
            translation = response.text.strip()
            
            # Remove any quotes that might be added
            if translation.startswith('"') and translation.endswith('"'):
                translation = translation[1:-1]
            
            print(f"Translation: '{text}' → '{translation}' ({target_lang})")
            return translation
            
        except Exception as e:
            print(f"Translation error: {e}")
            return text  # Return original text on error

# Global translator instance (will be initialized when needed)
_global_translator = None

def detect_language(text: str, api_key: str = None) -> str:
    """
    Detect the language of input text using Gemini AI
    
    Args:
        text: Text to analyze
        api_key: Ignored (for compatibility with old interface)
        
    Returns:
        Language code
    """
    global _global_translator
    
    if not GEMINI_AVAILABLE:
        print("Gemini AI not available, returning default language")
        return "en"
    
    try:
        if _global_translator is None:
            _global_translator = GeminiTranslator()
        return _global_translator.detect_language(text)
    except Exception as e:
        print(f"Language detection failed: {e}")
        return "en"

def translate_text(text: str, target: str, source: str = None, api_key: str = None) -> str:
    """
    Translate text using Gemini AI
    
    Args:
        text: Text to translate
        target: Target language code
        source: Source language code (optional)
        api_key: Ignored (for compatibility with old interface)
        
    Returns:
        Translated text
    """
    global _global_translator
    
    if not GEMINI_AVAILABLE:
        print("Gemini AI not available, returning original text")
        return text
    
    try:
        if _global_translator is None:
            _global_translator = GeminiTranslator()
        return _global_translator.translate_text(text, target, source)
    except Exception as e:
        print(f"Translation failed: {e}")
        return text

# Test function
# Test function removed - module is integrated into main system
