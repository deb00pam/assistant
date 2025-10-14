"""
Gemini AI Package

This package contains all Gemini AI-powered components for the assistant.

Components:
- client.py: Gemini AI client and configuration
- intent_classifier.py: 4-way intent classification
- local_retrieval.py: Local data retrieval with command generation
- web_retrieval.py: Web search and information retrieval
- translation.py: Language detection and translation
"""

# Import main components for easy access
try:
    from .client import GeminiClient
    from .intent_classifier import GeminiIntentClassifier
    from .local_retrieval import GeminiLocalDataRetriever
    from .web_retrieval import web_search
    from .translation import detect_language, translate_text
except ImportError:
    # Handle import errors gracefully
    pass

__all__ = [
    'GeminiClient',
    'GeminiIntentClassifier', 
    'GeminiLocalDataRetriever',
    'web_search',
    'detect_language',
    'translate_text'
]