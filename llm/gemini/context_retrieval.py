#!/usr/bin/env python3
"""
Gemini-Powered Context Retrieval System for Truvo
Uses Google's Gemini AI for intelligent conversation history retrieval and analysis.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import contextlib

# Suppress stderr temporarily for clean imports
@contextlib.contextmanager
def suppress_stderr():
    """Suppress stderr temporarily."""
    import sys
    with open(os.devnull, 'w') as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr

# Import Gemini with error suppression
with suppress_stderr():
    import google.generativeai as genai

from automation.storage import context_manager, ConversationTurn

@dataclass
class ContextResult:
    """Result from Gemini-powered context retrieval"""
    relevant_conversations: List[ConversationTurn]
    summary: str
    key_insights: List[str]
    confidence_score: float
    search_query: str

class GeminiContextRetriever:
    """Advanced context retrieval using Gemini AI"""

    def __init__(self, gemini_client=None):
        self.logger = logging.getLogger(__name__)

        # Use provided client or create new one
        if gemini_client:
            self.gemini_client = gemini_client
        else:
            # Import here to avoid circular imports
            try:
                from llm.gemini.client import GeminiClient
                self.gemini_client = GeminiClient()
            except ImportError:
                self.logger.error("Gemini client not available for context retrieval")
                self.gemini_client = None

        self.context_manager = context_manager

    def retrieve_context(self, query: str, max_results: int = 5) -> ContextResult:
        """
        Retrieve relevant conversation context using Gemini AI

        Args:
            query: The search query or current conversation topic
            max_results: Maximum number of relevant conversations to return

        Returns:
            ContextResult with relevant conversations, summary, and insights
        """
        if not self.gemini_client:
            # Fallback to basic retrieval
            basic_results = self.context_manager.get_relevant_context(query, max_results)
            return ContextResult(
                relevant_conversations=basic_results,
                summary="Basic keyword matching (Gemini unavailable)",
                key_insights=[],
                confidence_score=0.5,
                search_query=query
            )

        try:
            # Get recent conversations from database
            recent_conversations = self._get_recent_conversations(limit=50)

            if not recent_conversations:
                return ContextResult(
                    relevant_conversations=[],
                    summary="No conversation history available",
                    key_insights=[],
                    confidence_score=0.0,
                    search_query=query
                )

            # Use Gemini to analyze and find relevant context
            analysis_prompt = self._build_analysis_prompt(query, recent_conversations)
            analysis_response = self._call_gemini(analysis_prompt)

            # Parse Gemini's response
            parsed_result = self._parse_gemini_response(analysis_response, recent_conversations)

            return ContextResult(
                relevant_conversations=parsed_result["conversations"][:max_results],
                summary=parsed_result["summary"],
                key_insights=parsed_result["insights"],
                confidence_score=parsed_result["confidence"],
                search_query=query
            )

        except Exception as e:
            self.logger.error(f"Error in Gemini context retrieval: {e}")
            # Fallback to basic retrieval
            basic_results = self.context_manager.get_relevant_context(query, max_results)
            return ContextResult(
                relevant_conversations=basic_results,
                summary=f"Error in AI analysis: {str(e)}",
                key_insights=[],
                confidence_score=0.3,
                search_query=query
            )

    def get_conversation_insights(self, days_back: int = 7) -> Dict[str, Any]:
        """
        Get insights about conversation patterns using Gemini

        Args:
            days_back: Number of days to analyze

        Returns:
            Dictionary with insights, patterns, and recommendations
        """
        if not self.gemini_client:
            return {"error": "Gemini client not available"}

        try:
            # Get conversations from the specified period
            conversations = self._get_conversations_by_date(days_back)

            if not conversations:
                return {"insights": "No conversations found in the specified period"}

            # Build insights prompt
            insights_prompt = self._build_insights_prompt(conversations, days_back)
            insights_response = self._call_gemini(insights_prompt)

            # Parse insights response
            return self._parse_insights_response(insights_response)

        except Exception as e:
            self.logger.error(f"Error getting conversation insights: {e}")
            return {"error": f"Failed to analyze conversations: {str(e)}"}

    def find_similar_topics(self, current_topic: str) -> List[Dict[str, Any]]:
        """
        Find conversations with similar topics using semantic analysis

        Args:
            current_topic: The topic to find similar conversations for

        Returns:
            List of similar conversation topics with relevance scores
        """
        if not self.gemini_client:
            return []

        try:
            # Get all conversation topics/themes
            all_conversations = self._get_recent_conversations(limit=100)

            if len(all_conversations) < 5:
                return []

            # Use Gemini to find semantic similarities
            similarity_prompt = self._build_similarity_prompt(current_topic, all_conversations)
            similarity_response = self._call_gemini(similarity_prompt)

            return self._parse_similarity_response(similarity_response, all_conversations)

        except Exception as e:
            self.logger.error(f"Error finding similar topics: {e}")
            return []

    def _get_recent_conversations(self, limit: int = 50) -> List[ConversationTurn]:
        """Get recent conversations from the database"""
        try:
            cursor = self.context_manager.conn.cursor()
            cursor.execute('''
                SELECT timestamp, user_message, assistant_response, context_tags, importance_score
                FROM conversations
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,))

            return [ConversationTurn(*row) for row in cursor.fetchall()]
        except Exception as e:
            self.logger.error(f"Error getting recent conversations: {e}")
            return []

    def _get_conversations_by_date(self, days_back: int) -> List[ConversationTurn]:
        """Get conversations from the last N days"""
        try:
            cutoff_date = (datetime.now() - timedelta(days=days_back)).isoformat()
            cursor = self.context_manager.conn.cursor()
            cursor.execute('''
                SELECT timestamp, user_message, assistant_response, context_tags, importance_score
                FROM conversations
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
            ''', (cutoff_date,))

            return [ConversationTurn(*row) for row in cursor.fetchall()]
        except Exception as e:
            self.logger.error(f"Error getting conversations by date: {e}")
            return []

    def _build_analysis_prompt(self, query: str, conversations: List[ConversationTurn]) -> str:
        """Build the prompt for Gemini to analyze conversation relevance"""
        # Format conversations for Gemini
        conv_text = ""
        for i, conv in enumerate(conversations[:20]):  # Limit to avoid token limits
            conv_text += f"\nConversation {i+1}:\n"
            conv_text += f"User: {conv.user_message}\n"
            conv_text += f"Assistant: {conv.assistant_response}\n"
            conv_text += f"Time: {conv.timestamp}\n"
            if conv.context_tags:
                try:
                    tags = json.loads(conv.context_tags) if isinstance(conv.context_tags, str) else conv.context_tags
                    conv_text += f"Tags: {', '.join(tags)}\n"
                except:
                    pass
            conv_text += "---\n"

        return f"""You are an expert at analyzing conversation history to find relevant context.

CURRENT QUERY: "{query}"

CONVERSATION HISTORY:
{conv_text}

TASK: Analyze the conversation history and identify the most relevant conversations for the current query.

Return your analysis in this exact JSON format:
{{
    "relevant_indices": [1, 3, 5],
    "summary": "Brief summary of why these conversations are relevant",
    "key_insights": ["Insight 1", "Insight 2", "Insight 3"],
    "confidence_score": 0.85
}}

Guidelines:
- relevant_indices: Array of conversation numbers (1-based) that are most relevant
- summary: 1-2 sentences explaining the relevance
- key_insights: 2-4 key insights or patterns from relevant conversations
- confidence_score: Float between 0.0-1.0 indicating how confident you are

Focus on semantic meaning, not just keyword matches. Consider context, intent, and conversation flow."""

    def _build_insights_prompt(self, conversations: List[ConversationTurn], days_back: int) -> str:
        """Build prompt for conversation insights analysis"""
        # Format conversations for analysis
        conv_summary = ""
        for conv in conversations[:30]:  # Sample of conversations
            conv_summary += f"• {conv.user_message[:100]}... -> {conv.assistant_response[:100]}...\n"

        return f"""Analyze this conversation history from the last {days_back} days and provide insights.

CONVERSATIONS SAMPLE:
{conv_summary}

Provide insights in this JSON format:
{{
    "conversation_patterns": ["Pattern 1", "Pattern 2"],
    "frequent_topics": ["Topic 1", "Topic 2"],
    "user_behavior_insights": ["Insight 1", "Insight 2"],
    "recommendations": ["Recommendation 1", "Recommendation 2"],
    "overall_summary": "Brief summary of conversation trends"
}}

Focus on:
- Common conversation patterns
- Frequently discussed topics
- User behavior insights
- Suggestions for better interactions"""

    def _build_similarity_prompt(self, current_topic: str, conversations: List[ConversationTurn]) -> str:
        """Build prompt for finding similar conversation topics"""
        # Extract topics from recent conversations
        topics = []
        for conv in conversations[:20]:
            # Simple topic extraction - could be enhanced
            user_msg = conv.user_message.lower()
            if len(user_msg.split()) > 3:
                topics.append(user_msg[:150] + "...")

        topics_text = "\n".join(f"{i+1}. {topic}" for i, topic in enumerate(topics))

        return f"""Find conversations with topics similar to: "{current_topic}"

PAST CONVERSATION TOPICS:
{topics_text}

Return in JSON format:
{{
    "similar_topics": [
        {{"index": 1, "topic": "topic text", "similarity_score": 0.9}},
        {{"index": 3, "topic": "topic text", "similarity_score": 0.8}}
    ]
}}

Focus on semantic similarity, not just keyword matches."""

    def _call_gemini(self, prompt: str) -> str:
        """Call Gemini API with the given prompt"""
        try:
            with suppress_stderr():
                response = self.gemini_client.model.generate_content(prompt)
            return response.text if response and response.text else ""
        except Exception as e:
            self.logger.error(f"Gemini API call failed: {e}")
            return ""

    def _parse_gemini_response(self, response: str, conversations: List[ConversationTurn]) -> Dict[str, Any]:
        """Parse Gemini's JSON response for context analysis"""
        try:
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)

                # Get relevant conversations
                relevant_indices = data.get('relevant_indices', [])
                relevant_convs = []
                for idx in relevant_indices:
                    if 1 <= idx <= len(conversations):
                        relevant_convs.append(conversations[idx-1])

                return {
                    "conversations": relevant_convs,
                    "summary": data.get('summary', 'No summary provided'),
                    "insights": data.get('key_insights', []),
                    "confidence": data.get('confidence_score', 0.5)
                }
        except Exception as e:
            self.logger.error(f"Error parsing Gemini response: {e}")

        # Fallback
        return {
            "conversations": conversations[:3],  # Return first 3 as fallback
            "summary": "AI analysis failed, showing recent conversations",
            "insights": [],
            "confidence": 0.3
        }

    def _parse_insights_response(self, response: str) -> Dict[str, Any]:
        """Parse insights response from Gemini"""
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
        except Exception as e:
            self.logger.error(f"Error parsing insights response: {e}")

        return {"error": "Failed to parse insights"}

    def _parse_similarity_response(self, response: str, conversations: List[ConversationTurn]) -> List[Dict[str, Any]]:
        """Parse similarity analysis response"""
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)

                similar_topics = data.get('similar_topics', [])
                results = []
                for item in similar_topics:
                    idx = item.get('index', 0)
                    if 1 <= idx <= len(conversations):
                        conv = conversations[idx-1]
                        results.append({
                            "conversation": conv,
                            "similarity_score": item.get('similarity_score', 0.0),
                            "topic": item.get('topic', '')
                        })
                return results
        except Exception as e:
            self.logger.error(f"Error parsing similarity response: {e}")

        return []

# Global instance
gemini_context_retriever = GeminiContextRetriever()