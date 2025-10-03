#!/usr/bin/env python3
"""
Gemini Intent Classifier - AI-powered intent detection using Google's Gemini

This module uses Gemini AI to classify user queries into 4 intent types:
- conversation: chat, questions, greetings, casual talk
- automation: desktop actions, file operations, system commands
- local_data_retrieval: searching local files, documents, system info
- web_data_retrieval: web searches, online information lookup

Benefits over traditional ML approach:
- Better context understanding
- Handles ambiguous cases with 4-way classification
- No training data needed
- More natural language processing
"""

import os
import sys
from typing import Optional, Dict, Any
import json

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Use existing GeminiClient infrastructure
try:
    import sys
    import os
    # Add the parent directory to Python path to ensure imports work
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from .client import GeminiClient
    from core.config import AssistantConfig
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError as e:
    print(f"Import error: {e}")
    GEMINI_AVAILABLE = False

class GeminiIntentClassifier:
    """Intent classifier using Gemini AI for more intelligent classification"""
    
    def __init__(self, model_name: str = "auto", gemini_client=None):
        """
        Initialize the Gemini Intent Classifier
        
        Args:
            model_name: Gemini model to use ('auto' for best available)
            gemini_client: Existing GeminiClient instance to reuse (optional)
        """
        if not GEMINI_AVAILABLE:
            raise ImportError("GeminiClient not available. Check llm.gemini_client import")
        
        try:
            if gemini_client is not None:
                # Use the provided GeminiClient instance (shares same model as assistant)
                self.gemini_client = gemini_client
                self.model = gemini_client.model
                self.model_name = "shared_with_assistant"
            else:
                # Create our own GeminiClient instance
                config = AssistantConfig(model_name=model_name)
                self.gemini_client = GeminiClient(config=config)
                self.model = self.gemini_client.model
                self.model_name = model_name
        except Exception as e:
            print(f"Error initializing Gemini Intent Classifier: {e}")
            raise
    
    def classify_intent(self, query: str) -> str:
        """
        Classify user intent using Gemini AI into 4 categories
        
        Args:
            query: User's input text
            
        Returns:
            str: 'conversation', 'automation', 'local_data_retrieval', or 'web_data_retrieval'
        """
        if not query or not query.strip():
            return 'conversation'
        
        # Create a detailed prompt for 4-way classification
        prompt = f"""You are an AI assistant that helps classify user intents into 4 specific categories.

Analyze the following user query and determine which category it belongs to:

1. CONVERSATION - Casual chat, greetings, general questions, discussions that don't require data or actions
2. AUTOMATION - Requests to perform actions on the computer (opening apps, clicking buttons, file operations, system commands, taking screenshots, etc.)
3. LOCAL_DATA_RETRIEVAL - Searching for information in local files, documents, or system (find files, read documents, check system info, search local data)
4. WEB_DATA_RETRIEVAL - Searching for information online, web searches, looking up current information (weather, news, definitions, online research)

User Query: "{query}"

Classification Guidelines:
- "open Chrome", "click button", "take screenshot" = AUTOMATION
- "hello", "how are you", "tell me a joke" = CONVERSATION  
- "find my resume", "what's in my documents", "search local files" = LOCAL_DATA_RETRIEVAL
- "what's the weather", "search for Python tutorials", "latest news" = WEB_DATA_RETRIEVAL

Respond with ONLY one word: CONVERSATION, AUTOMATION, LOCAL_DATA_RETRIEVAL, or WEB_DATA_RETRIEVAL
"""

        try:
            # Generate response from Gemini using the model directly
            response = self.model.generate_content(prompt)
            result_text = response.text.strip().upper()
            
            # Parse the response into the 4 categories
            if "AUTOMATION" in result_text:
                intent = "automation"
            elif "LOCAL_DATA_RETRIEVAL" in result_text:
                intent = "local_data_retrieval"
            elif "WEB_DATA_RETRIEVAL" in result_text:
                intent = "web_data_retrieval"
            elif "CONVERSATION" in result_text:
                intent = "conversation"
            else:
                # Fallback: if unclear response, default to conversation for safety
                print(f"Unclear Gemini response: '{result_text}', defaulting to conversation")
                intent = "conversation"
            
            # Log the classification (optional debug info)
            if os.getenv('DEBUG_INTENT', '').lower() == 'true':
                print(f"Gemini classified '{query}' as {intent}")
            
            return intent
            
        except Exception as e:
            print(f"Gemini classification error: {e}")
            print("Falling back to conversation for safety")
            return "conversation"  # Default to conversation if Gemini fails
    
    def classify_with_confidence(self, query: str) -> Dict[str, Any]:
        """
        Get detailed classification with reasoning for all 4 intent types
        
        Args:
            query: User's input text
            
        Returns:
            dict: Contains intent, confidence, and reasoning
        """
        if not query or not query.strip():
            return {
                "intent": "conversation",
                "confidence": 1.0,
                "reasoning": "Empty query"
            }
        
        # Enhanced prompt with reasoning request for 4-way classification
        prompt = f"""You are an AI assistant that classifies user intents with detailed analysis into 4 categories.

Analyze this user query: "{query}"

Classify as one of these 4 types:
1. CONVERSATION - Casual chat, greetings, general questions, discussions
2. AUTOMATION - Computer actions (open apps, click, file ops, system commands)  
3. LOCAL_DATA_RETRIEVAL - Search local files, documents, system info
4. WEB_DATA_RETRIEVAL - Web searches, online information lookup

Respond in this EXACT JSON format:
{{
    "classification": "CONVERSATION" or "AUTOMATION" or "LOCAL_DATA_RETRIEVAL" or "WEB_DATA_RETRIEVAL",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation of why"
}}

Examples:
- "open Chrome" → AUTOMATION (clear app launch command)
- "hello there" → CONVERSATION (greeting)
- "find my resume" → LOCAL_DATA_RETRIEVAL (search local files)
- "what's the weather today" → WEB_DATA_RETRIEVAL (online information)
- "search for Python files" → LOCAL_DATA_RETRIEVAL (local file search)
- "google machine learning" → WEB_DATA_RETRIEVAL (web search)
"""

        try:
            response = self.model.generate_content(prompt)
            result_text = response.text.strip()
            
            # Try to parse JSON response
            try:
                # Extract JSON from response (in case there's extra text)
                start = result_text.find('{')
                end = result_text.rfind('}') + 1
                if start >= 0 and end > start:
                    json_str = result_text[start:end]
                    result = json.loads(json_str)
                    
                    # Parse classification
                    classification = result.get("classification", "CONVERSATION").upper()
                    intent_map = {
                        "CONVERSATION": "conversation",
                        "AUTOMATION": "automation", 
                        "LOCAL_DATA_RETRIEVAL": "local_data_retrieval",
                        "WEB_DATA_RETRIEVAL": "web_data_retrieval"
                    }
                    intent = intent_map.get(classification, "conversation")
                    
                    return {
                        "intent": intent,
                        "confidence": float(result.get("confidence", 0.8)),
                        "reasoning": result.get("reasoning", "Gemini classification"),
                        "raw_response": result_text
                    }
                    
            except json.JSONDecodeError:
                # Fallback: parse simple text response
                result_upper = result_text.upper()
                if "AUTOMATION" in result_upper:
                    intent = "automation"
                elif "LOCAL_DATA_RETRIEVAL" in result_upper:
                    intent = "local_data_retrieval"
                elif "WEB_DATA_RETRIEVAL" in result_upper:
                    intent = "web_data_retrieval"
                else:
                    intent = "conversation"
                    
                return {
                    "intent": intent,
                    "confidence": 0.7,
                    "reasoning": "Parsed from text response",
                    "raw_response": result_text
                }
                
        except Exception as e:
            print(f"Gemini detailed classification error: {e}")
            return {
                "intent": "conversation",
                "confidence": 0.5,
                "reasoning": f"Error: {e}",
                "raw_response": str(e)
            }

# Test function removed - module is integrated into main system