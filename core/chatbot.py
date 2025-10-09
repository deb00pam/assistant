#!/usr/bin/env python3
"""
ChatBot Module for Truvo Desktop Assistant
Handles conversational interactions using Gemini AI with multilingual support.
"""

# Set up environment if this module is imported directly
try:
    from core.config import setup_environment
    setup_environment()
except ImportError:
    pass  # main.py will handle it

import os
import re
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import contextlib
import sys

# Import translation utilities - now powered by Gemini AI!
try:
    from llm.gemini.translation import detect_language, translate_text
    TRANSLATION_AVAILABLE = True
except Exception as e:
    print(f"Translation not available in chatbot: {e}")
    TRANSLATION_AVAILABLE = False
    # Translation now handled by llm.gemini.translation module

# Import user data utilities - REMOVED: using context_memory.db instead
# from automation.storage import load_user_data, save_user_data

# Load user data at startup - REMOVED: using context_memory.db instead
# USER_DATA = load_user_data()

# Suppress stderr temporarily
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

# Google AI imports with error suppression
with suppress_stderr():
    import google.generativeai as genai

# Note: Intent classification is now handled by GeminiIntentClassifier in main.py
# Old intent classifiers (PostgreSQL-based) are no longer used

# Import GeminiClient from llm package
try:
    from llm.gemini.client import GeminiClient
except ImportError:
    GeminiClient = None

# Import Gemini Context Retrieval
try:
    from llm.gemini.context_retrieval import GeminiContextRetriever
    GEMINI_CONTEXT_AVAILABLE = True
except ImportError:
    GeminiContextRetriever = None
    GEMINI_CONTEXT_AVAILABLE = False

# Import context manager for database storage
from automation.storage import context_manager

# Try to import OS detection
try:
    from automation.os_detection import get_os_context
    OS_DETECTION_AVAILABLE = True
except ImportError:
    get_os_context = None
    OS_DETECTION_AVAILABLE = False

# Legacy data modules removed - now using Gemini AI
# from data.local_retrieval import universal_assistant  # Now using llm/gemini_local_retrieval.py
# from data.web_retrieval import web_retrieval          # Now using llm/gemini_web_retrieval.py
LOCAL_DATA_AVAILABLE = True   # Gemini local retrieval always available
WEB_DATA_AVAILABLE = True     # Gemini web retrieval always available

# Combined availability flag
DATA_RETRIEVAL_AVAILABLE = LOCAL_DATA_AVAILABLE or WEB_DATA_AVAILABLE


class ChatBot:
    """Conversational chatbot using Gemini AI for natural language interactions."""
    
    def __init__(self, gemini_client):
        self.gemini_client = gemini_client
        self.conversation_history = []
        self.max_history_length = 20  # Keep last 20 exchanges
        
        # Note: Intent classification is now handled by GeminiIntentClassifier in main.py
        # No need for the old PostgreSQL-based classifier
        self.intent_classifier = None
        
        self._pending_search_query = None  # Store pending browser searches
        
        # Initialize data retrieval capabilities
        self.data_retrieval_available = DATA_RETRIEVAL_AVAILABLE
        self.local_data_available = LOCAL_DATA_AVAILABLE
        self.web_data_available = WEB_DATA_AVAILABLE
        
        # Initialize Gemini-powered context retrieval
        if GEMINI_CONTEXT_AVAILABLE:
            self.context_retriever = GeminiContextRetriever(gemini_client)
        else:
            self.context_retriever = None
        
        if self.local_data_available:
            # Legacy universal_assistant removed - using Gemini AI
            self.universal_assistant = None
        else:
            self.universal_assistant = None
            
        if self.web_data_available:
            # Legacy web_retrieval removed - using Gemini AI
            self.web_retrieval = None
        else:
            self.web_retrieval = None
        
        # System prompt for the chatbot personality with OS context
        os_context = ""
        if OS_DETECTION_AVAILABLE and get_os_context:
            try:
                os_context = f"\n\nSYSTEM ENVIRONMENT: {get_os_context()}\n"
                os_context += "IMPORTANT: When providing commands, file paths, or technical instructions, use the appropriate format for this operating system.\n"
            except Exception as e:
                os_context = ""
        
        self.system_prompt = f"""You are a helpful and friendly AI assistant integrated into Truvo, a desktop automation tool. 
        You can have natural conversations about any topic while also being aware that you're part of a system 
        that can perform desktop automation tasks.{os_context}
        
        CURRENT DATE: {datetime.now().strftime("%B %d, %Y")} (Remember this for any date-related queries)
        
        When users ask general questions, provide helpful and engaging responses. 
        
        When users ask about your capabilities or what tasks you can perform, explain that you can:
        1. Have natural conversations on any topic
        2. Perform desktop automation tasks like:
           - Opening applications (browsers, notepad, calculator, etc.)
           - Clicking buttons and UI elements
           - Typing text and pressing keys
           - Taking screenshots and analyzing screens
           - Navigating websites and software
        3. Retrieve comprehensive data including:
           - SYSTEM INFO: Memory, disk space, PC name, uptime, hardware details
           - SYSTEM COMMANDS: File counts, directory listings, system operations
           - LIVE WEB DATA: Stock prices, weather, news, sports scores (via LangChain)
           - WEB SEARCH: Professional-grade web search using DuckDuckGo and SerpAPI
           - DIRECT APIS: Weather data, financial data, news feeds, sports information
           
        IMPORTANT: I have integrated local and web data retrieval capabilities.
        - For system queries (like "free memory", "PC name"), I use local system data
        - For live data (like "Tesla stock price", "weather"), I use advanced web search with LangChain
        - I provide sources and confidence scores for web-retrieved information
        
        CRITICAL INSTRUCTION: When I provide you with data retrieval results (marked with [Local Data Retrieved] or [Web Data Retrieved]), you MUST:
        1. Use ONLY the information provided in those results
        2. Do NOT add any information not present in the search results
        3. Do NOT make assumptions or fill in gaps with made-up information
        4. If the search results are incomplete or unclear, say so honestly
        5. Always acknowledge the sources provided and stick to the facts found
           
        If someone asks "can you perform a task for me?" or similar, respond conversationally by asking 
        what specific task they have in mind, rather than immediately trying to automate something.
        
        Be friendly, helpful, and engaging in all interactions."""
    
    # def save_user_state(self): - REMOVED: using context_memory.db instead
    #     """Save relevant user data."""
    #     USER_DATA['last_conversation_history'] = self.conversation_history
    #     USER_DATA['last_used'] = datetime.now().isoformat()
    #     save_user_data(USER_DATA)
    
    def _handle_browser_request(self, user_message: str) -> str:
        """Handle user request to open browser."""
        # Check if user is agreeing to open browser
        agreement_words = ['yes', 'yeah', 'ok', 'okay', 'sure', 'go ahead', 'do it', 'open it', 'search it', 'please']
        user_lower = user_message.lower().strip()
        
        if any(word in user_lower for word in agreement_words) and len(user_lower) < 20:  # Short responses only
            # Look for pending search query in recent conversation history
            search_query = None
            for message in reversed(self.conversation_history[-3:]):  # Check last 3 messages
                if message.get('role') == 'assistant':
                    content = message.get('content', '')
                    if 'BROWSER_SEARCH_PENDING:' in content:
                        # Extract the search query
                        parts = content.split('BROWSER_SEARCH_PENDING:')
                        if len(parts) > 1:
                            search_query = parts[1].strip()
                            break
            
            if search_query:
                # Legacy universal_assistant browser functionality removed
                # Using Gemini AI for search instead
                return f"I understand you want to search for '{search_query}'. Please open your browser and search manually, or use the web data retrieval feature."
        
        return ""  # Not a browser request

    def chat(self, user_message: str) -> str:
        """Process a conversational message and return a response with multilingual support."""
        api_key = os.getenv('GOOGLE_TRANSLATE_API_KEY')
        config = getattr(self, 'config', None)
        user_lang = getattr(config, 'user_language', 'auto') if config else 'auto'

        # Always detect user language for every input
        try:
            detected_lang = detect_language(user_message, api_key)
        except Exception:
            detected_lang = 'en'

        # Let Gemini handle all language detection intelligently

        # Always translate user message to English for Gemini
        processed_message = user_message
        if detected_lang != 'en':
            try:
                processed_message = translate_text(user_message, target='en', source=detected_lang, api_key=api_key)
                print(f"[DEBUG] User input translated to English: {processed_message}")
            except Exception:
                processed_message = user_message
                print(f"[DEBUG] User input translation failed, using original: {processed_message}")

        # Check for special commands
        if processed_message.lower().strip() in ['clear history', 'clear chat', 'reset conversation', 'new conversation']:
            self.conversation_history = []
            return translate_text("Conversation history cleared! We're starting fresh. 🧹", target=detected_lang, api_key=api_key) if detected_lang != 'en' else "Conversation history cleared! We're starting fresh. 🧹"

        # Check if this is a response to a browser suggestion first
        browser_response = self._handle_browser_request(processed_message)
        if browser_response:
            return translate_text(browser_response, target=detected_lang, api_key=api_key) if detected_lang != 'en' else browser_response

        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message,
            "timestamp": datetime.now().isoformat()
        })
        # self.save_user_state() - REMOVED: using context_memory.db instead

        # Build conversation context
        context = self._build_conversation_context(processed_message)

        # Check if user query needs data retrieval
        retrieval_context = ""
        if self.data_retrieval_available:
            retrieval_context = self._handle_data_retrieval(processed_message)

        # Add retrieval context to the prompt if available
        if retrieval_context:
            context += f"\n\nREAL-TIME DATA CONTEXT:\n{retrieval_context}\n"

        # Generate response using Gemini, retrying with other models if quota/rate limit error occurs
        tried_models = set()
        last_error = None
        available_models = self.gemini_client.list_available_models() if hasattr(self.gemini_client, 'list_available_models') else []
        # Always try the current model first
        current_model = getattr(self.gemini_client.model, 'model_name', None) or getattr(self.gemini_client.model, 'name', None)
        if current_model:
            available_models = [current_model] + [m for m in available_models if m != current_model]
        for model_name in available_models:
            try:
                if model_name != current_model:
                    # Switch to new model
                    with suppress_stderr():
                        self.gemini_client.model = genai.GenerativeModel(model_name)
                with suppress_stderr():
                    response = self.gemini_client.model.generate_content(context)
                if response and response.text:
                    bot_response = response.text.strip()
                    # Clean up any hidden markers from the response
                    if 'BROWSER_SEARCH_PENDING:' in bot_response:
                        bot_response = bot_response.split('BROWSER_SEARCH_PENDING:')[0].strip()
                    # Add bot response to history (store both context and clean response)
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": context if 'BROWSER_SEARCH_PENDING:' in context else bot_response,
                        "response": bot_response,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # Save conversation to database for context retrieval
                    try:
                        context_manager.save_conversation_turn(
                            user_msg=user_message,
                            assistant_response=bot_response,
                            context_tags=[],  # Could be enhanced to extract topics
                            importance=1.0   # Could be enhanced with importance scoring
                        )
                    except Exception as e:
                        # Don't fail if database save fails
                        pass
                    self._trim_history()
                    # self.save_user_state() - REMOVED: using context_memory.db instead
                    # Let Gemini handle all language detection and translation intelligently
                    if detected_lang and detected_lang != 'en':
                        print(f"[DEBUG] Translating Gemini reply to: {detected_lang}")
                        try:
                            translated = translate_text(bot_response, target=detected_lang, source='en', api_key=api_key)
                            print(f"[DEBUG] Translated response: {translated}")
                            return translated
                        except Exception as ex:
                            print(f"[DEBUG] Translation failed: {ex}")
                            return bot_response
                    return bot_response
                else:
                    last_error = "I'm sorry, I couldn't generate a response right now. Please try again."
                    break
            except Exception as e:
                error_msg = str(e)
                last_error = error_msg
                if any(keyword in error_msg.lower() for keyword in ['429', 'quota', 'rate limit', 'requests per']):
                    tried_models.add(model_name)
                    continue  # Try next model
                else:
                    logging.error(f"Chat error: {e}")
                    return f"I encountered an error: {error_msg}"
        # If all models exhausted or only quota errors
        if last_error and any(keyword in last_error.lower() for keyword in ['429', 'quota', 'rate limit', 'requests per']):
            msg = ("**API Quota Exceeded**\n\n"
                   "I've hit the daily request limit for all available models. Please try:\n"
                   "• Wait 24 hours for quota reset\n"
                   "• Use a paid Google AI API key for higher limits\n\n"
                   f"Technical details: {last_error[:100]}...")
            if detected_lang and detected_lang != 'en':
                print(f"[DEBUG] Translating error message to: {detected_lang}")
                try:
                    return translate_text(msg, target=detected_lang, api_key=api_key)
                except Exception as ex:
                    print(f"[DEBUG] Translation failed: {ex}")
                    return msg
            return msg
        elif last_error:
            if detected_lang and detected_lang != 'en':
                print(f"[DEBUG] Translating error message to: {detected_lang}")
                try:
                    return translate_text(f"I encountered an error: {last_error}", target=detected_lang, api_key=api_key)
                except Exception as ex:
                    print(f"[DEBUG] Translation failed: {ex}")
                    return f"I encountered an error: {last_error}"
            return f"I encountered an error: {last_error}"
        else:
            fallback_msg = "I'm sorry, I couldn't generate a response right now. Please try again."
            if detected_lang and detected_lang != 'en':
                print(f"[DEBUG] Translating fallback message to: {detected_lang}")
                try:
                    return translate_text(fallback_msg, target=detected_lang, api_key=api_key)
                except Exception as ex:
                    print(f"[DEBUG] Translation failed: {ex}")
                    return fallback_msg
            return fallback_msg
    
    def _build_conversation_context(self, current_query: str = "") -> str:
        """Build the conversation context for Gemini using AI-powered retrieval."""
        context_parts = [self.system_prompt]

        # Use Gemini-powered context retrieval if available
        if self.context_retriever and current_query:
            try:
                context_result = self.context_retriever.retrieve_context(current_query, max_results=8)

                if context_result.relevant_conversations:
                    context_parts.append("\n\nRelevant Conversation History:")
                    for conv in context_result.relevant_conversations:
                        context_parts.append(f"User: {conv.user_message}")
                        context_parts.append(f"Assistant: {conv.assistant_response}")

                    # Add context summary if available
                    if context_result.summary and context_result.summary != "Basic keyword matching (Gemini unavailable)":
                        context_parts.append(f"\nContext Summary: {context_result.summary}")

                else:
                    # Fallback to recent history
                    context_parts.append("\n\nRecent Conversation History:")
                    for entry in self.conversation_history[-6:]:  # Last 6 exchanges as fallback
                        role = "Human" if entry["role"] == "user" else "Assistant"
                        context_parts.append(f"{role}: {entry['content']}")

            except Exception as e:
                # Fallback to basic recent history
                context_parts.append("\n\nRecent Conversation History:")
                for entry in self.conversation_history[-6:]:
                    role = "Human" if entry["role"] == "user" else "Assistant"
                    context_parts.append(f"{role}: {entry['content']}")
        else:
            # Fallback when no AI context retrieval available
            context_parts.append("\n\nRecent Conversation History:")
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
    
    # Legacy context analysis removed - Gemini handles context intelligently
    
    def _needs_web_search(self, query: str) -> bool:
        """Determine if a query needs web search using enhanced intent classification."""
        # Use enhanced intent classifier if available
        if ENHANCED_INTENT_AVAILABLE and isinstance(self.intent_classifier, EnhancedIntentClassifier):
            try:
                result = self.intent_classifier.classify_intent(query)
                return result.get('needs_web_search', False)
            except Exception as e:
                print(f"Intent classification failed: {e}")
        
        # Fallback to keyword-based detection
        query_lower = query.lower()
        
        # Let Gemini determine intelligently if web search is needed
        return True  # Gemini will handle the decision

    def _handle_data_retrieval(self, user_message: str) -> str:
        """Simplified data retrieval - let Gemini handle everything intelligently."""
        if not self.data_retrieval_available:
            return ""
            
        try:
            # Gemini AI now handles all data retrieval intelligently
            # Legacy complex formatting removed - Gemini processes raw data
            return f"[Data context for: '{user_message}' - Gemini will handle retrieval intelligently]"
                
        except Exception as e:
            return f"[Data retrieval error: {str(e)}]"
    
    # Legacy data formatting methods removed - Gemini handles everything intelligently
