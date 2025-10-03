#!/usr/bin/env python3
"""
Intent Classifier Module for Truvo Desktop Assistant
NLP-based intent classifier to distinguish between chat and automation tasks.
"""

# NLP and ML imports
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: NLP libraries not installed. Using basic keyword detection. Install with: pip install scikit-learn nltk")

# Import the intent training database
try:
    from intent_database import IntentTrainingDatabase, get_default_postgresql_config
    DATABASE_AVAILABLE = True
except ImportError:
    try:
        from .intent_database import IntentTrainingDatabase, get_default_postgresql_config
        DATABASE_AVAILABLE = True
    except ImportError:
        DATABASE_AVAILABLE = False


class IntentClassifier:
    """NLP-based intent classifier to distinguish between chat and automation tasks."""
    
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.is_trained = False
        
        if SKLEARN_AVAILABLE:
            self._initialize_nltk()
            self._train_classifier()
        else:
            print("Using fallback keyword-based detection (install scikit-learn for better accuracy)")
    
    def _initialize_nltk(self):
        """Initialize NLTK resources."""
        try:
            import ssl
            try:
                _create_unverified_https_context = ssl._create_unverified_context
            except AttributeError:
                pass
            else:
                ssl._create_default_https_context = _create_unverified_https_context
            
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
        except Exception as e:
            print(f"Warning: NLTK setup warning: {e}")
    
    def _preprocess_text(self, text):
        """Preprocess text for classification."""
        try:
            # Tokenize and clean
            tokens = word_tokenize(text.lower())
            
            # Remove stopwords
            stop_words = set(stopwords.words('english'))
            filtered_tokens = [w for w in tokens if w not in stop_words and w.isalpha()]
            
            # Lemmatize
            lemmatizer = WordNetLemmatizer()
            lemmatized = [lemmatizer.lemmatize(token) for token in filtered_tokens]
            
            return ' '.join(lemmatized)
        except Exception:
            # Fallback to simple preprocessing
            return text.lower().strip()
    
    def _train_classifier(self):
        """Train the intent classification model using PostgreSQL database."""
        try:
            # Get training data from PostgreSQL database
            if not DATABASE_AVAILABLE:
                raise ImportError("Intent database not available. Install database dependencies.")
            
            # PostgreSQL configuration
            postgresql_config = get_default_postgresql_config()
            
            db = IntentTrainingDatabase(postgresql_config)
            training_data = db.get_training_data()
            print(f"Loaded {len(training_data)} training examples from PostgreSQL database")
            
            if not training_data:
                raise ValueError("No training data available in database")
            
            # Separate texts and labels
            texts = [self._preprocess_text(text) for text, label in training_data]
            labels = [label for text, label in training_data]
            
            # Create and train the model
            self.model = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=1000, ngram_range=(1, 2))),
                ('classifier', MultinomialNB(alpha=0.1))
            ])
            
            self.model.fit(texts, labels)
            self.is_trained = True
            
            # Train classifier silently
            accuracy = self.model.score(texts, labels)
            
        except Exception as e:
            print(f"Training failed: {e}")
            self.is_trained = False
            raise

    
    def classify_intent(self, text: str) -> bool:
        """
        Classify if text is conversational (True) or automation task (False).
        Requires trained NLP model from PostgreSQL database.
        """
        if not text.strip():
            return True  # Empty text is conversational
        
        if not (SKLEARN_AVAILABLE and self.is_trained):
            raise RuntimeError("Intent classifier not properly trained. Check PostgreSQL database connection.")
        
        try:
            processed_text = self._preprocess_text(text)
            prediction = self.model.predict([processed_text])[0]
            probabilities = self.model.predict_proba([processed_text])[0]
            
            # Get confidence scores
            automation_confidence = probabilities[0]
            conversation_confidence = probabilities[1]
            max_confidence = max(automation_confidence, conversation_confidence)
            
            # Apply confidence threshold - if confidence is too low, default to conversation
            CONFIDENCE_THRESHOLD = 0.7  # 70% confidence required
            
            if max_confidence < CONFIDENCE_THRESHOLD:
                return True  # Default to conversation for ambiguous cases
            
            return prediction == 1  # 1 = conversation, 0 = automation
            
        except Exception as e:
            print(f"Classification failed: {e}")
            raise



if __name__ == "__main__":
    # Test the intent classifier
    classifier = IntentClassifier()
    
    test_queries = [
        "open chrome browser",
        "hello how are you", 
        "show disk space",
        "trending news today",
        "what is machine learning",
        "like why me always bro"
    ]
    
    print("\nTesting Intent Classification:")
    for query in test_queries:
        result = classifier.classify_intent(query)
        intent_type = "conversation" if result else "automation"
        print(f"'{query}' -> {intent_type}")
