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

# Import translation utilities
from ai.translation import detect_language, translate_text

# Import user data utilities
from data.storage import load_user_data, save_user_data

# Load user data at startup
USER_DATA = load_user_data()

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

# Import IntentClassifier directly from ai.intent_classifier
try:
    from ai.intent_classifier import IntentClassifier
except ImportError:
    IntentClassifier = None

# Import GeminiClient from llm package
try:
    from llm.gemini_client import GeminiClient
except ImportError:
    GeminiClient = None

# Try to import OS detection
try:
    from utils.os_detection import get_os_context
    OS_DETECTION_AVAILABLE = True
except ImportError:
    get_os_context = None
    OS_DETECTION_AVAILABLE = False

# Try to import data retrieval modules
try:
    from data.local_retrieval import universal_assistant
    LOCAL_DATA_AVAILABLE = True
except ImportError:
    universal_assistant = None
    LOCAL_DATA_AVAILABLE = False

try:
    from data.web_retrieval import web_retrieval
    WEB_DATA_AVAILABLE = True
except ImportError:
    web_retrieval = None
    WEB_DATA_AVAILABLE = False

# Combined availability flag
DATA_RETRIEVAL_AVAILABLE = LOCAL_DATA_AVAILABLE or WEB_DATA_AVAILABLE


class ChatBot:
    """Conversational chatbot using Gemini AI for natural language interactions."""
    
    def __init__(self, gemini_client):
        self.gemini_client = gemini_client
        self.conversation_history = []
        self.max_history_length = 20  # Keep last 20 exchanges
        self.intent_classifier = IntentClassifier() if IntentClassifier else None
        self._pending_search_query = None  # Store pending browser searches
        
        # Initialize data retrieval capabilities
        self.data_retrieval_available = DATA_RETRIEVAL_AVAILABLE
        self.local_data_available = LOCAL_DATA_AVAILABLE
        self.web_data_available = WEB_DATA_AVAILABLE
        
        if self.local_data_available:
            self.universal_assistant = universal_assistant
        else:
            self.universal_assistant = None
            
        if self.web_data_available:
            self.web_retrieval = web_retrieval
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
    
    def save_user_state(self):
        """Save relevant user data."""
        USER_DATA['last_conversation_history'] = self.conversation_history
        USER_DATA['last_used'] = datetime.now().isoformat()
        save_user_data(USER_DATA)
    
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
                # Open browser
                if self.universal_assistant:
                    browser_opened = self.universal_assistant.open_browser_search(search_query)
                    if browser_opened:
                        return f"Great! I've opened your browser and searched for '{search_query}'. You should see the results now."
                    else:
                        return f"Sorry, I couldn't open the browser automatically. Please search for '{search_query}' manually."
                else:
                    return f"Please search for '{search_query}' manually in your browser."
        
        return ""  # Not a browser request

    def chat(self, user_message: str) -> str:
        """Process a conversational message and return a response with multilingual support."""
        api_key = os.getenv('GOOGLE_TRANSLATE_API_KEY')
        config = getattr(self, 'config', None)
        user_lang = getattr(config, 'user_language', 'auto') if config else 'auto'

        # Always detect user language for every input
        try:
            detected_lang = detect_language(user_message, api_key)
            print(f"[DEBUG] Detected user language: {detected_lang}")
        except Exception:
            detected_lang = 'en'
            print("[DEBUG] Language detection failed, defaulting to 'en'")

        # Heuristic for transliterated Bengali/Hindi (Benglish/Hinglish)
        def is_benglish(text):
            # Common Bengali words written in English
            bengali_keywords = ['bhalo', 'achis', 'tor', 'amar', 'kemon', 'ki', 'tui', 'korchis', 'kothay', 'khub', 'shob', 'shotti', 'bondhu', 'dost', 'pagol', 'khub', 'khushi', 'shobai']
            return any(word in text.lower() for word in bengali_keywords)

        def is_hinglish(text):
            # Common Hindi words written in English
            hindi_keywords = ['hai', 'kya', 'kaise', 'tum', 'mera', 'tera', 'acha', 'bura', 'dost', 'pyaar', 'shukriya', 'namaste', 'bhai', 'bahut', 'accha', 'kyun', 'kyon', 'sab', 'theek']
            return any(word in text.lower() for word in hindi_keywords)

        # If language detection says English but it's actually Ben/Hinglish, override
        if detected_lang == 'en':
            if is_benglish(user_message):
                detected_lang = 'benglish'
                print('[DEBUG] Heuristic: Detected BenGlish (Bengali in English script)')
            elif is_hinglish(user_message):
                detected_lang = 'hinglish'
                print('[DEBUG] Heuristic: Detected Hinglish (Hindi in English script)')

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
        self.save_user_state()

        # Build conversation context
        context = self._build_conversation_context()

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
                    self._trim_history()
                    self.save_user_state()
                    # Always reply in BenGlish (English script Bengali) if user input is BenGlish or Bengali
                    if detected_lang in ['bn', 'benglish'] or is_benglish(user_message):
                        print(f"[DEBUG] Forcing BenGlish output (English script Bengali) regardless of detected_lang")
                        try:
                            from indic_transliteration.sanscript import transliterate
                            from indic_transliteration.sanscript import BENGALI, ITRANS
                            bengali_text = translate_text(bot_response, target='bn', source='en', api_key=api_key)
                            benglish = transliterate(bengali_text, BENGALI, ITRANS)
                            print(f"[DEBUG] BenGlish transliteration: {benglish}")
                            return benglish
                        except Exception as ex:
                            print(f"[DEBUG] BenGlish transliteration failed: {ex}")
                            return bot_response
                    elif detected_lang in ['hi', 'hinglish'] or is_hinglish(user_message):
                        print(f"[DEBUG] Always replying in Hinglish style (English script Hindi)")
                        try:
                            from indic_transliteration.sanscript import transliterate
                            from indic_transliteration.sanscript import DEVANAGARI, ITRANS
                            hindi_text = translate_text(bot_response, target='hi', source='en', api_key=api_key)
                            hinglish = transliterate(hindi_text, DEVANAGARI, ITRANS)
                            print(f"[DEBUG] Hinglish transliteration: {hinglish}")
                            return hinglish
                        except Exception as ex:
                            print(f"[DEBUG] Hinglish transliteration failed: {ex}")
                            return bot_response
                    elif detected_lang and detected_lang != 'en':
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
            msg = ("🚫 **API Quota Exceeded**\n\n"
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
    
    def _build_conversation_context(self) -> str:
        """Build the conversation context for Gemini."""
        context_parts = [self.system_prompt, "\n\nConversation History:"]
        
        # Add recent conversation history
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
    
    def is_conversational_query(self, text: str) -> bool:
        """Determine if text is a conversational query vs desktop automation command using NLP."""
        if self.intent_classifier:
            return self.intent_classifier.classify_intent(text)
        return True  # Default to conversational if no classifier
    
    def _analyze_conversational_context(self, user_message: str) -> str:
        """Analyze recent conversation history to understand references and context."""
        message_lower = user_message.lower()
        
        # Look for pronoun references that need context
        pronoun_patterns = [
            r'\bthey\b', r'\bthem\b', r'\btheir\b', r'\bit\b', r'\bthis\b', r'\bthat\b',
            r'\bhe\b', r'\bshe\b', r'\bhim\b', r'\bher\b', r'\bhis\b', r'\bhers\b',
            r'\bthere\b'
        ]
        
        has_pronouns = any(re.search(pattern, message_lower) for pattern in pronoun_patterns)
        
        if has_pronouns:
            # Look back through recent conversation history for context
            entities = []
            
            # Check last 10 messages for ANY entity references
            recent_messages = self.conversation_history[-10:] if len(self.conversation_history) >= 10 else self.conversation_history
            
            for msg in recent_messages:
                content = msg.get('content', '').lower()
                
                # Look for team names (prioritize actual club names)
                football_clubs = [
                    'manchester city', 'man city', 'arsenal', 'liverpool', 
                    'chelsea', 'tottenham', 'spurs', 'united', 'manchester united',
                    'barcelona', 'real madrid', 'psg', 'bayern', 'dortmund'
                ]
                
                national_teams = [
                    'india', 'pakistan', 'australia', 'england', 'south africa'
                ]
                
                teams = football_clubs + national_teams
                
                # Look for places/locations
                places = [
                    'london', 'paris', 'new york', 'tokyo', 'delhi', 'mumbai',
                    'berlin', 'madrid', 'rome', 'moscow', 'beijing', 'sydney',
                    'los angeles', 'chicago', 'boston', 'seattle', 'miami',
                    'france', 'germany', 'italy', 'spain', 'japan', 'china',
                    'uk', 'usa', 'america', 'canada', 'australia', 'brazil'
                ]
                
                # Look for people/celebrities
                people = [
                    'messi', 'ronaldo', 'neymar', 'mbappe', 'haaland',
                    'biden', 'trump', 'modi', 'putin', 'xi jinping',
                    'elon musk', 'bill gates', 'jeff bezos', 'mark zuckerberg'
                ]
                
                # Look for companies/brands
                companies = [
                    'apple', 'google', 'microsoft', 'amazon', 'meta', 'facebook',
                    'tesla', 'netflix', 'spotify', 'uber', 'airbnb', 'twitter', 'x'
                ]
                
                # Look for movies/shows/books
                entertainment = [
                    'marvel', 'disney', 'netflix', 'hbo', 'game of thrones',
                    'avengers', 'batman', 'superman', 'star wars', 'harry potter'
                ]
                
                # Combine all entity lists (longer names first to avoid partial matches)
                all_entities = sorted(teams + places + people + companies + entertainment, key=len, reverse=True)
                
                for entity in all_entities:
                    if entity in content and entity not in entities and len(entity) > 1:
                        entities.append(entity)
            
            # If we found entities and the user is asking about "them/they/it/this/that"
            if entities:
                # Prioritize based on context and pronouns
                latest_entity = entities[-1]  # Default to most recent
                
                # Sports context: prioritize teams over countries when asking about "playing/matches"
                if re.search(r'\b(they|them|their)\b', message_lower) and re.search(r'\b(playing|match|game|fixtures|next)\b', message_lower):
                    # First look for football clubs, then national teams
                    club_entities = [e for e in entities if e in football_clubs]
                    if club_entities:
                        latest_entity = club_entities[-1]  # Most recent club mentioned
                    else:
                        national_entities = [e for e in entities if e in national_teams]
                        if national_entities:
                            latest_entity = national_entities[-1]
                
                # People context: prefer people if asking about "him/her"
                elif re.search(r'\b(he|she|him|her)\b', message_lower):
                    person_entities = [e for e in entities if e in people]
                    if person_entities:
                        latest_entity = person_entities[-1]
                        
                # Location context: prefer cities if asking about "there"
                elif re.search(r'\bthere\b', message_lower):
                    place_entities = [e for e in entities if e in places]
                    if place_entities:
                        # Prefer specific cities over countries
                        city_entities = [e for e in place_entities if e not in ['france', 'germany', 'italy', 'spain', 'japan', 'china', 'uk', 'usa', 'america', 'england', 'australia', 'brazil']]
                        latest_entity = city_entities[-1] if city_entities else place_entities[-1]
                
                # Common question patterns that benefit from context
                context_patterns = [
                    r'when.*?(playing|happening|coming|starting|releasing)',
                    r'what.*?(score|news|update|latest|new)',
                    r'where.*?(located|based|from|playing)',
                    r'how.*?(doing|performing|much|many)',
                    r'tell me.*?(more|about|latest)',
                    r'(latest|recent|new|current).*?(news|update|info)'
                ]
                
                needs_context = any(re.search(pattern, message_lower) for pattern in context_patterns)
                
                if needs_context:
                    enhanced_query = f"{latest_entity} {user_message}"
                    return enhanced_query
        
        return user_message

    def _handle_data_retrieval(self, user_message: str) -> str:
        """Handle ANY question using integrated Local + Web retrieval system."""
        if not self.data_retrieval_available:
            return ""
            
        try:
            local_result = None
            web_result = None
            
            # Step 1: Try local data retrieval first (system, commands, conversational)
            if self.local_data_available and self.universal_assistant:
                local_result = self.universal_assistant.answer_anything(user_message)
                
                # Check if local retrieval was successful and sufficient
                if local_result.get('success'):
                    query_type = local_result.get('query_type', 'unknown')
                    
                    # For system, command, and conversational queries, use local result
                    if query_type in ['system', 'command', 'conversational']:
                        local_result['source'] = 'local'
                        return self._format_unified_context(local_result, user_message)
            
            # Step 2: Try web retrieval for live data, factual queries, or when local fails
            if self.web_data_available and self.web_retrieval:
                try:
                    web_result = self.web_retrieval.get_answer_from_web(user_message)
                    
                    if web_result.get('success'):
                        web_result['source'] = 'web'
                        web_result['query_type'] = 'web_search'
                        return self._format_unified_context(web_result, user_message)
                        
                except Exception as e:
                    print(f"⚠️ Web retrieval failed: {e}")
            
            # Step 3: Fallback to local result if web failed
            if local_result and local_result.get('success'):
                local_result['source'] = 'local_fallback'
                return self._format_unified_context(local_result, user_message)
            
            # Step 4: Ultimate fallback
            return f"[Could not retrieve information for: '{user_message}'. Both local and web retrieval unavailable or failed.]"
                
        except Exception as e:
            return f"[Data retrieval error: {str(e)}]"
    
    def _format_unified_context(self, result: Dict[str, Any], original_query: str) -> str:
        """Format result from either local or web retrieval into context for AI."""
        query_type = result.get('query_type', 'unknown')
        answer = result.get('answer', '')
        source = result.get('source', 'unknown')
        
        # Build context header based on source
        if source == 'local':
            context_header = f"[LOCAL DATA RETRIEVED FOR: '{original_query}']"
        elif source == 'web':
            context_header = f"[WEB DATA RETRIEVED FOR: '{original_query}']"
            # Add source information for web results
            sources = result.get('sources', [])
            if sources:
                source_count = len(sources)
                context_header += f" (Sources: {source_count})"
        else:
            context_header = f"[DATA RETRIEVED FOR: '{original_query}' (Source: {source})]"
        
        # Start with clear instruction to AI
        context_parts = [
            context_header,
            "INSTRUCTION: Use ONLY the information below. Do not add any details not present in this data.",
            "=" * 60
        ]
        
        # Add the answer with clear labeling
        if answer:
            context_parts.append(f"RETRIEVED INFORMATION:")
            context_parts.append(answer)
            context_parts.append("=" * 60)
        
        # Add source URLs for web results with clear labeling
        if source == 'web' and result.get('sources'):
            context_parts.append("SOURCES:")
            for i, s in enumerate(result.get('sources', [])[:3], 1):
                if s.get('url') and s.get('url') != 'https://duckduckgo.com':
                    context_parts.append(f"{i}. {s.get('title', 'Unknown')} - {s.get('url', 'No URL')}")
            context_parts.append("=" * 60)
        
        # Final instruction
        context_parts.append("REMINDER: Answer using ONLY the information above. Do not speculate or add extra details.")
        
        return "\n".join(context_parts)
    
    def _format_retrieval_context(self, result: Dict[str, Any], original_query: str) -> str:
        """Format retrieval result into context for AI."""
        method = result.get('method_used', 'unknown')
        browser_fallback = result.get('browser_fallback', False)
        
        context_parts = [f"[Retrieved data for query: '{original_query}' using method: {method}]"]
        
        # Handle universal browser fallback first
        if browser_fallback:
            fallback_reason = result.get('fallback_reason', 'Data not satisfactory')
            search_query = result.get('search_query', original_query)
            
            # Store the search query for later use (needs to be passed somehow)
            # For now, we'll include it in the context
            context_parts.append(f"Data Retrieval: {fallback_reason}.")
            context_parts.append(f"Would you like me to open your browser and search for '{search_query}' to get better information? (Just say 'yes' or 'ok')")
            context_parts.append(f"BROWSER_SEARCH_PENDING:{search_query}")  # Hidden marker for the system
            
            # Don't automatically open browser - let the user decide
            return "\n".join(context_parts)
        
        # Continue with normal data formatting only if no browser fallback
        if method == "web":
            title = result.get('title', 'No title')
            text_content = result.get('text_content', '')[:500]  # Limit length
            context_parts.append(f"Page Title: {title}")
            if text_content:
                context_parts.append(f"Content Preview: {text_content}")
        
        elif method == "command":
            stdout = result.get('stdout', '')[:500]  # Limit length
            if stdout:
                context_parts.append(f"Command Output: {stdout}")
            
            stderr = result.get('stderr', '')
            if stderr:
                context_parts.append(f"Command Error: {stderr}")
        
        elif method == "search":
            # Check for sports-specific data first
            if result.get('sports_data'):
                sports_data = result['sports_data']
                if sports_data.get('summary'):
                    context_parts.append(f"Sports Info: {sports_data['summary']}")
                if sports_data.get('match_details'):
                    context_parts.append(f"Match Details: {sports_data['match_details']}")
            
            # Check for Google search results
            elif result.get('search_results'):
                context_parts.append(f"Found {len(result['search_results'])} search results")
                if result.get('top_result_content'):
                    content = result['top_result_content'].get('text_content', '')[:300]
                    if content:
                        context_parts.append(f"Top Result: {content}")
            
            # Fallback to standard search results
            else:
                abstract = result.get('abstract', '')
                if abstract:
                    context_parts.append(f"Search Summary: {abstract}")
                
                instant_answer = result.get('instant_answer', '')
                if instant_answer:
                    context_parts.append(f"Quick Answer: {instant_answer}")
        
        elif method == "api":
            # Weather data
            if 'weather' in original_query.lower():
                weather = result.get('current_weather', {})
                if weather:
                    temp_c = weather.get('temperature_c', 'N/A')
                    condition = weather.get('condition', 'N/A')
                    location = result.get('location', 'Unknown location')
                    context_parts.append(f"Weather in {location}: {temp_c}°C, {condition}")
            
            # Sports data
            elif any(sport in original_query.lower() for sport in ['cricket', 'match', 'score', 'live']):
                has_live_data = result.get('has_live_data', False)
                verified_live = result.get('metadata', {}).get('verified_live', False)
                browser_fallback = result.get('browser_fallback', False)
                
                if not has_live_data:
                    if browser_fallback:
                        # Offer to open browser
                        search_query = result.get('search_query', original_query)
                        context_parts.append(f"Sports Search: No live matches found for '{original_query}'. Opening your browser to search for the latest information...")
                        
                        # Open browser with search
                        if hasattr(self, 'universal_assistant') and self.universal_assistant:
                            browser_opened = self.universal_assistant.open_browser_search(f"{search_query} live score cricket")
                            if browser_opened:
                                context_parts.append("Browser opened with your sports search query.")
                            else:
                                context_parts.append("Could not open browser automatically. Please search manually for live cricket scores.")
                        else:
                            context_parts.append("Please check sports websites manually for live match information.")
                    else:
                        context_parts.append(f"Sports Search: No live matches found for '{original_query}'. The requested match may not be currently playing.")
                elif not verified_live:
                    abstract = result.get('abstract', '')
                    context_parts.append(f"Sports Info: Found some data but cannot verify if it's currently live: {abstract[:200]}")
                else:
                    # Only show data we've verified as live
                    if result.get('sports_data'):
                        sports_info = result['sports_data']
                        context_parts.append(f"Live Sports Update: {sports_info.get('summary', 'Live match information found')}")
                    elif result.get('abstract'):
                        context_parts.append(f"Live Sports Update: {result.get('abstract', '')}")
                    else:
                        context_parts.append(f"Live Sports: Match data found but format unclear")
        
        # Add metadata
        timestamp = result.get('timestamp', '')
        if timestamp:
            context_parts.append(f"Retrieved at: {timestamp}")
        
        return "\n".join(context_parts)
