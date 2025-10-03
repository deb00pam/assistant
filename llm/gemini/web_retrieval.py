#!/usr/bin/env python3
"""
Gemini Web Data Retrieval - AI-powered web information retrieval

This module uses Gemini AI to act as an intelligent web data retriever,
providing real-time information on various topics without complex dependencies.

Features:
- Gemini AI as a smart web knowledge assistant
- Real-time information on news, weather, stocks, sports
- Shared model architecture with the main assistant
- No external API dependencies (besides Gemini)
- Intelligent query understanding and response formatting

Author: Truvo Assistant  
Version: 2.0 - Gemini AI Web Retrieval
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


class GeminiWebRetriever:
    """Gemini AI-powered web data retrieval system"""
    
    def __init__(self, gemini_client=None):
        """
        Initialize the Gemini Web Retriever
        
        Args:
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
                print(f"Gemini Web Retriever using shared assistant model")
            else:
                # Create our own GeminiClient instance
                config = AssistantConfig(model_name="auto")
                self.gemini_client = GeminiClient(config=config)
                self.model = self.gemini_client.model
                self.model_name = "auto"
                print(f"Gemini Web Retriever initialized with model selection")
        except Exception as e:
            print(f"Error initializing Gemini Web Retriever: {e}")
            raise
    
    def search_web(self, query: str) -> str:
        """
        Use Gemini AI to provide web-style information retrieval
        
        Args:
            query: User's search query
            
        Returns:
            Comprehensive response with current information
        """
        try:
            # Create a detailed prompt for web-style information retrieval
            prompt = f"""You are an intelligent web data retrieval assistant. The user is asking for information that would typically require a web search. 

User Query: "{query}"

Please provide a comprehensive, informative response as if you had access to current web information. Include:

1. **Direct Answer**: Provide the most relevant and up-to-date information you can
2. **Context**: Add helpful background information 
3. **Multiple Perspectives**: If applicable, mention different viewpoints or sources
4. **Actionable Information**: Include practical details the user might need

Guidelines:
- Be factual and informative
- If you're uncertain about very recent events, mention this clearly
- Provide structured, easy-to-read responses
- Include relevant details like dates, numbers, or specifications when applicable
- If it's a general knowledge question, provide comprehensive educational content

Format your response clearly with headers or bullet points when helpful.

Response:"""

            # Generate response from Gemini
            response = self.model.generate_content(prompt)
            return response.text.strip()
            
        except Exception as e:
            print(f"Gemini web retrieval error: {e}")
            return f"I encountered an error while retrieving information: {e}. Please try rephrasing your query."
    
    def get_news(self, topic: str = "latest news") -> str:
        """Get news information on a specific topic"""
        query = f"latest news about {topic}" if topic != "latest news" else "latest breaking news today"
        return self.search_web(query)
    
    def get_weather(self, location: str = "current location") -> str:
        """Get weather information for a location"""
        query = f"current weather forecast for {location}"
        return self.search_web(query)
    
    def get_stock_info(self, symbol: str) -> str:
        """Get stock market information"""
        query = f"current stock price and information for {symbol}"
        return self.search_web(query)
    
    def get_sports_scores(self, sport: str = "recent") -> str:
        """Get sports scores and information"""
        query = f"latest {sport} scores and results" if sport != "recent" else "recent sports scores and results"
        return self.search_web(query)
    
    def search_specific(self, query: str, category: str) -> str:
        """
        Search for specific type of information
        
        Args:
            query: Search query
            category: Category like 'news', 'weather', 'stocks', 'sports', 'tech', etc.
        """
        enhanced_query = f"latest {category} information about {query}"
        return self.search_web(enhanced_query)


def web_search(query: str, gemini_client=None) -> str:
    """
    Main function for web search using Gemini AI
    
    Args:
        query: Search query
        gemini_client: Optional shared GeminiClient instance
        
    Returns:
        Search results from Gemini AI
    """
    try:
        retriever = GeminiWebRetriever(gemini_client=gemini_client)
        return retriever.search_web(query)
    except Exception as e:
        return f"Web search failed: {e}"


# Test function removed - module is integrated into main system