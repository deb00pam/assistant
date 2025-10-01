"""
AI Module - Artificial Intelligence components for Truvo

This module contains AI-powered features including chatbot, intent classification, and translation.
"""

# Lazy imports to avoid circular dependencies
def __getattr__(name):
    if name == 'ChatBot':
        from .chatbot import ChatBot
        return ChatBot
    elif name == 'IntentClassifier':
        from .intent_classifier import IntentClassifier
        return IntentClassifier
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = ['ChatBot', 'IntentClassifier']
