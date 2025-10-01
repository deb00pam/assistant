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
            print("🔄 Using fallback keyword-based detection (install scikit-learn for better accuracy)")
    
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
        """Train the intent classification model."""
        try:
            # Training data: (text, label) where 0=automation, 1=conversation
            training_data = [
                    # Automation tasks (0)
                    ("open chrome browser", 0),
                    ("click on the button", 0),
                    ("type hello world", 0),
                    ("press enter key", 0),
                    ("scroll down page", 0),
                    ("launch notepad application", 0),
                    ("close window", 0),
                    ("minimize application", 0),
                    ("take screenshot", 0),
                    ("open file explorer", 0),
                    ("start calculator", 0),
                    ("run program", 0),
                    ("navigate to website", 0),
                    ("download file", 0),
                    ("save document", 0),
                    ("copy text", 0),
                    ("paste clipboard", 0),
                    ("delete file", 0),
                    ("install software", 0),
                    ("update application", 0),
                    ("switch to tab", 0),
                    ("maximize window", 0),
                    ("drag and drop", 0),
                    ("select all text", 0),
                    ("go to url", 0),
                    ("open edge", 0),
                    ("start spotify", 0),
                    ("launch terminal", 0),
                    ("execute command", 0),
                    ("find text", 0),
                    ("can you open edge", 0),
                    ("can you open chrome", 0),
                    ("open edge for me", 0),
                    ("like can you open edge", 0),
                    ("please open chrome", 0),
                    ("launch notepad please", 0),
                    ("start calculator for me", 0),
                    ("can you click the button", 0),
                    ("please type this text", 0),
                    ("can you scroll down", 0),
                    ("help me open firefox", 0),
                    ("open the browser", 0),
                    ("start the application", 0),
                    ("run the program", 0),
                    ("execute this command", 0),
                    ("launch spotify app", 0),
                    ("open file manager", 0),
                    ("start windows explorer", 0),
                    # Local data fetch (2)
                    ("show disk space", 2),
                    ("how much memory is free", 2),
                    ("what is the pc name", 2),
                    ("system uptime", 2),
                    ("list files in downloads", 2),
                    ("show cpu usage", 2),
                    ("get hardware info", 2),
                    ("show battery status", 2),
                    ("run command dir", 2),
                    ("get windows version", 2),
                    ("show network info", 2),
                    ("get ip address", 2),
                    ("show running processes", 2),
                    ("show system info", 2),
                    ("get local user accounts", 2),
                    ("show available drives", 2),
                    ("get bios version", 2),
                    ("show ram details", 2),
                    ("get motherboard info", 2),
                    # Web data fetch (deep research) (3)
                    ("what is the weather in kolkata", 3),
                    ("tesla stock price today", 3),
                    ("latest news headlines", 3),
                    ("who won the cricket match", 3),
                    ("bitcoin price now", 3),
                    ("show me trending topics", 3),
                    ("covid cases in india", 3),
                    ("compare m3 vs ryzen 7 battery life reviews", 3),
                    ("causes and fixes for windows 11 random freezes", 3),
                    ("what happened at openai this year", 3),
                    ("find python tutorials", 3),
                    ("search for best laptops 2025", 3),
                    ("show me live football scores", 3),
                    ("weather in london right now", 3),
                    ("tesla share price", 3),
                    ("who is the president of france", 3),
                    ("how tall is mount everest", 3),
                    ("show me recent earthquakes", 3),
                    ("find latest science discoveries", 3),
                    ("show me apple earnings report", 3),
                    ("find top movies this week", 3),
                    ("show me sports news", 3),
                    ("find github trending repos", 3),
                    ("show me live weather radar", 3),
                    ("find best restaurants near me", 3),
                    # Conversational queries (1)
                    ("hello", 1),
                    ("hi", 1),
                    ("hey", 1),
                    ("hello there", 1),
                    ("hi there", 1),
                    ("hey there", 1),
                    ("good morning", 1),
                    ("good afternoon", 1),
                    ("good evening", 1),
                    ("how are you", 1),
                    ("what is your name", 1),
                    ("tell me a joke", 1),
                    ("how do you work", 1),
                    ("what can you do", 1),
                    ("explain machine learning", 1),
                    ("what do you think about", 1),
                    ("are you intelligent", 1),
                    ("do you have feelings", 1),
                    ("thanks for helping", 1),
                    ("you are awesome", 1),
                    ("good job well done", 1),
                    ("i love you", 1),
                    ("how was your day", 1),
                    ("tell me about science", 1),
                    ("why is sky blue", 1),
                    ("what time is it", 1),
                    ("who invented computer", 1),
                    ("recommend a movie", 1),
                    ("i feel sad today", 1),
                    ("you make me happy", 1),
                    ("goodbye see you later", 1),
                    ("hi there", 1),
                    ("hey buddy", 1),
                    ("thank you so much", 1),
                    ("amazing work", 1),
                    ("that was great", 1),
                    ("i think you are", 1),
                    ("do you believe in", 1),
                    ("what is meaning of life", 1),
                    ("tell me story", 1),
                    ("sing a song", 1),
                    ("make me laugh", 1),
                    ("i need emotional support", 1),
                    ("you are my friend", 1),
                    ("how old are you", 1),
                    ("where are you from", 1),
                    ("what are your hobbies", 1),
                    ("do you dream", 1),
                    ("are you lonely", 1),
                    ("what makes you happy", 1),
                    ("hiii", 1),
                    ("yeppppp", 1),
                    ("yeahhh", 1),
                    ("niceee", 1),
                    ("coollll", 1),
                    ("can you help me with something", 1),
                    ("what do you think about this", 1),
                    ("i want to ask you something", 1),
                    ("are you there", 1),
                    ("how smart are you", 1),
                    ("do you understand me", 1),
                    ("can you perform a task for me", 1),
                    ("what task can you do", 1),
                    ("what kind of tasks", 1),
                    ("can you do work for me", 1),
                    ("help me with a task", 1),
                    ("i need help with task", 1),
                    ("what automation can you do", 1),
                    ("like batman is the prime example", 1),
                    ("batman sacrificed everything", 1),
                    ("like in the movie", 1),
                    ("like the character", 1),
                    ("this reminds me of", 1),
                ]
            
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
            print(f"Warning: NLP training failed: {e}. Using fallback detection.")
            self.is_trained = False
    
    def classify_intent(self, text: str) -> bool:
        """
        Classify if text is conversational (True) or automation task (False).
        Uses NLP if available, otherwise falls back to keyword detection.
        """
        if not text.strip():
            return True  # Empty text is conversational
        
        if SKLEARN_AVAILABLE and self.is_trained:
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
                print(f"Warning: NLP classification failed: {e}. Using fallback.")
                return self._fallback_detection(text)
        else:
            return self._fallback_detection(text)
    
    def _fallback_detection(self, text: str) -> bool:
        """Fallback keyword-based detection when NLP is not available."""
        text_lower = text.lower().strip()
        
        # Clear automation commands
        automation_patterns = [
            'open ', 'launch ', 'start ', 'run ', 'execute ',
            'click', 'press ', 'type ', 'scroll', 'drag',
            'close ', 'minimize ', 'maximize ', 'switch to',
            'go to', 'navigate', 'download', 'save', 'delete',
            'copy', 'paste', 'install', 'screenshot'
        ]
        
        # Website/URL patterns - these are likely automation tasks
        website_patterns = [
            '.com', '.org', '.net', '.edu', '.gov', '.io',
            'google ', 'youtube', 'facebook', 'twitter', 'instagram',
            'website', 'browser', 'search ', 'www.'
        ]
        
        for pattern in automation_patterns:
            if pattern in text_lower:
                return False
                
        for pattern in website_patterns:
            if pattern in text_lower:
                return False  # Websites are automation tasks
        
        # Clear conversational patterns
        conversational_patterns = [
            text_lower in ['hi', 'hello', 'hey', 'thanks', 'bye'],
            text_lower.startswith(('what', 'how', 'why', 'who', 'when', 'where')),
            '?' in text,
            any(phrase in text_lower for phrase in [
                'tell me', 'explain', 'you are', 'i am', 'i feel',
                'love you', 'thank you', 'good job', 'awesome'
            ])
        ]
        
        return any(conversational_patterns)
