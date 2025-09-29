#!/usr/bin/env python3
"""
NLP Intent Classifier for Truvo

This module uses advanced NLP techniques to classify user queries into different
intent categories: conversational, system, command, live_data, factual, and web_search.

Uses sentence transformers for semantic understanding instead of keyword matching.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

# Try to import NLP libraries
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import pandas as pd
    HAS_NLP = True
except ImportError:
    HAS_NLP = False

class NLPIntentClassifier:
    """
    Advanced NLP-based intent classification using sentence transformers.
    
    Categories:
    - conversational: Greetings, emotions, personal chat
    - system: System information queries (memory, CPU, etc.)
    - command: System commands (file operations, etc.)
    - live_data: Real-time data (stocks, weather, news)
    - factual: General knowledge questions
    - web_search: Complex queries requiring web search
    """
    
    def __init__(self):
        self.model = None
        self.intent_examples = {}
        self.intent_embeddings = {}
        self.threshold = 0.6  # Minimum similarity threshold
        
        if HAS_NLP:
            self._initialize_model()
            self._setup_training_data()
            self._compute_embeddings()
            print("🧠 NLP Intent Classifier initialized with sentence transformers!")
        else:
            print("⚠️ NLP libraries not available. Install with: pip install sentence-transformers scikit-learn")
    
    def _initialize_model(self):
        """Initialize the sentence transformer model."""
        try:
            # Use a lightweight but effective model
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Sentence transformer model loaded")
        except Exception as e:
            print(f"⚠️ Failed to load sentence transformer: {e}")
            self.model = None
    
    def _setup_training_data(self):
        """Setup training examples for each intent category."""
        self.intent_examples = {
            'conversational': [
                "hello", "hi there", "how are you", "good morning",
                "i am sad", "i am lonely", "i feel happy", "thank you",
                "will you listen", "talk to me", "i need someone to chat",
                "goodbye", "see you later", "nice to meet you",
                "are you serious", "bro why", "what the hell",
                "i am excited", "feeling great today", "having a bad day",
                "can we talk", "i need to vent", "tell me something funny"
            ],
            'system': [
                "how much memory is available", "what is my pc name", "system information",
                "free memory", "disk space", "cpu usage", "computer specs",
                "what operating system", "username", "hostname", "uptime",
                "hardware info", "system stats", "computer details",
                "memory usage", "available ram", "storage space",
                "what version of windows", "machine name", "computer name"
            ],
            'command': [
                "how many files in folder", "list files", "directory contents",
                "count files", "show directory", "file count in downloads",
                "what files are here", "folder size", "directory listing",
                "number of files", "files in this directory",
                "disk usage of folder", "size of directory"
            ],
            'live_data': [
                "tesla stock price", "weather in london", "latest news",
                "bitcoin price", "current temperature", "sports scores",
                "breaking news", "stock market today", "weather forecast",
                "live updates", "real time data", "current prices",
                "today's news", "recent updates", "happening now"
            ],
            'factual': [
                "what is artificial intelligence", "how does photosynthesis work",
                "who invented the telephone", "explain quantum physics",
                "what is the capital of france", "define machine learning",
                "how tall is mount everest", "when was the internet invented",
                "what causes earthquakes", "history of world war 2"
            ],
            'web_search': [
                "best restaurants in paris", "how to bake chocolate cake",
                "reviews of iphone 15", "compare laptops under 1000",
                "durga puja pandals in kolkata", "travel guide to japan",
                "what to do in new york", "best movies of 2025",
                "learning python programming", "job opportunities in tech"
            ]
        }
    
    def _compute_embeddings(self):
        """Compute embeddings for all training examples."""
        if not self.model:
            return
            
        for intent, examples in self.intent_examples.items():
            try:
                embeddings = self.model.encode(examples)
                self.intent_embeddings[intent] = embeddings
                print(f"✅ Computed embeddings for {intent}: {len(examples)} examples")
            except Exception as e:
                print(f"⚠️ Failed to compute embeddings for {intent}: {e}")
    
    def classify_intent(self, query: str) -> Dict[str, any]:
        """
        Classify user query intent using NLP.
        
        Args:
            query: User's input query
            
        Returns:
            Dict with intent, confidence, and reasoning
        """
        if not HAS_NLP or not self.model:
            return self._fallback_classification(query)
        
        try:
            # Encode the query
            query_embedding = self.model.encode([query])
            
            # Calculate similarities with each intent
            intent_scores = {}
            for intent, embeddings in self.intent_embeddings.items():
                similarities = cosine_similarity(query_embedding, embeddings)
                max_similarity = np.max(similarities)
                avg_similarity = np.mean(similarities)
                
                # Combined score: weight max similarity more, but consider average
                combined_score = (max_similarity * 0.7) + (avg_similarity * 0.3)
                intent_scores[intent] = {
                    'max_similarity': float(max_similarity),
                    'avg_similarity': float(avg_similarity),
                    'combined_score': float(combined_score)
                }
            
            # Find the best intent
            best_intent = max(intent_scores.keys(), key=lambda x: intent_scores[x]['combined_score'])
            best_score = intent_scores[best_intent]['combined_score']
            
            # Apply threshold
            if best_score < self.threshold:
                # If no clear intent, classify based on query characteristics
                return self._heuristic_classification(query, intent_scores)
            
            return {
                'intent': best_intent,
                'confidence': best_score,
                'method': 'nlp_similarity',
                'all_scores': intent_scores,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"⚠️ NLP classification failed: {e}")
            return self._fallback_classification(query)
    
    def _heuristic_classification(self, query: str, scores: Dict) -> Dict[str, any]:
        """Apply heuristic rules when NLP confidence is low."""
        query_lower = query.lower()
        
        # Short queries are likely conversational
        if len(query.strip()) <= 3:
            return {
                'intent': 'conversational',
                'confidence': 0.8,
                'method': 'heuristic_short_query',
                'reasoning': 'Short queries are typically conversational'
            }
        
        # Questions starting with specific words
        question_patterns = {
            'what is': 'factual',
            'who is': 'factual', 
            'when is': 'factual',
            'how to': 'web_search',
            'where to': 'web_search',
            'best': 'web_search'
        }
        
        for pattern, intent in question_patterns.items():
            if query_lower.startswith(pattern):
                return {
                    'intent': intent,
                    'confidence': 0.75,
                    'method': 'heuristic_pattern',
                    'pattern': pattern
                }
        
        # Default to the highest NLP score even if below threshold
        best_intent = max(scores.keys(), key=lambda x: scores[x]['combined_score'])
        best_score = scores[best_intent]['combined_score']
        
        return {
            'intent': best_intent,
            'confidence': best_score,
            'method': 'nlp_low_confidence',
            'reasoning': 'Best NLP match despite low confidence'
        }
    
    def _fallback_classification(self, query: str) -> Dict[str, any]:
        """Fallback to keyword-based classification when NLP is unavailable."""
        query_lower = query.lower()
        
        # Simple keyword-based classification
        if any(word in query_lower for word in ['hello', 'hi', 'sad', 'lonely', 'feel', 'i am']):
            return {'intent': 'conversational', 'confidence': 0.7, 'method': 'keyword_fallback'}
        elif any(word in query_lower for word in ['memory', 'pc', 'system', 'computer']):
            return {'intent': 'system', 'confidence': 0.7, 'method': 'keyword_fallback'}
        elif any(word in query_lower for word in ['files', 'folder', 'directory']):
            return {'intent': 'command', 'confidence': 0.7, 'method': 'keyword_fallback'}
        elif any(word in query_lower for word in ['stock', 'weather', 'news', 'price']):
            return {'intent': 'live_data', 'confidence': 0.7, 'method': 'keyword_fallback'}
        else:
            return {'intent': 'web_search', 'confidence': 0.5, 'method': 'keyword_fallback_default'}

# Singleton instance
_nlp_classifier = None

def get_intent_classifier() -> NLPIntentClassifier:
    """Get or create the NLP intent classifier instance."""
    global _nlp_classifier
    if _nlp_classifier is None:
        _nlp_classifier = NLPIntentClassifier()
    return _nlp_classifier

def classify_user_intent(query: str) -> str:
    """
    Simple function to classify user intent and return just the intent string.
    
    Args:
        query: User's input query
        
    Returns:
        Intent category string
    """
    classifier = get_intent_classifier()
    result = classifier.classify_intent(query)
    return result.get('intent', 'web_search')

# Test function
if __name__ == "__main__":
    # Test the classifier
    classifier = NLPIntentClassifier()
    
    test_queries = [
        "hello how are you",
        "i am very sad today", 
        "what is my computer name",
        "how many files in downloads folder",
        "tesla stock price right now",
        "what is artificial intelligence",
        "best restaurants in tokyo"
    ]
    
    print("\n🧪 Testing NLP Intent Classification:")
    print("=" * 60)
    
    for query in test_queries:
        result = classifier.classify_intent(query)
        print(f"Query: '{query}'")
        print(f"Intent: {result['intent']} (confidence: {result['confidence']:.3f})")
        print(f"Method: {result['method']}")
        print("-" * 40)