#!/usr/bin/env python3
"""
Enhanced Intent Classification Database

This module creates and manages a PostgreSQL database for training data
to improve intent classification accuracy.
"""

import os
from typing import List, Tuple, Dict

def get_default_postgresql_config() -> Dict[str, str]:
    """Get default PostgreSQL configuration from environment variables."""
    return {
        'host': os.getenv('POSTGRES_HOST', 'localhost'),
        'port': os.getenv('POSTGRES_PORT', '5432'),
        'database': os.getenv('POSTGRES_DB', 'truvo_intent'),
        'user': os.getenv('POSTGRES_USER', 'truvo'),
        'password': os.getenv('POSTGRES_PASSWORD', 'password')
    }

# PostgreSQL imports
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    POSTGRESQL_AVAILABLE = True
except ImportError:
    POSTGRESQL_AVAILABLE = False
    raise ImportError("PostgreSQL support required. Install with: pip install psycopg2-binary")

class IntentTrainingDatabase:
    """PostgreSQL database for storing and managing intent classification training data."""
    
    def __init__(self, db_config: Dict = None):
        """Initialize the PostgreSQL database."""
        if not POSTGRESQL_AVAILABLE:
            raise ImportError("PostgreSQL support required. Install with: pip install psycopg2-binary")
        
        # PostgreSQL configuration
        self.db_config = db_config or get_default_postgresql_config()
        
        print("Using PostgreSQL for intent training database")
        
        # Initialize database
        self.init_database()
        self.populate_training_data()
    
    def init_database(self):
        """Create the PostgreSQL database tables."""
        try:
            with psycopg2.connect(**self.db_config) as conn:
                with conn.cursor() as cursor:
                    # Create training data table
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS training_data (
                            id SERIAL PRIMARY KEY,
                            text TEXT NOT NULL,
                            intent INTEGER NOT NULL,
                            intent_name TEXT NOT NULL,
                            confidence REAL DEFAULT 1.0,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    ''')
                    
                    # Create intent categories table
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS intent_categories (
                            id INTEGER PRIMARY KEY,
                            name TEXT NOT NULL,
                            description TEXT NOT NULL
                        )
                    ''')
                    
                    # Insert intent categories
                    categories = [
                        (0, 'automation', 'Desktop automation tasks like opening apps, clicking buttons'),
                        (1, 'conversation', 'General chat, greetings, philosophical questions'),
                        (2, 'system_info', 'Local system information queries'),
                        (3, 'web_search', 'Real-time information that needs web search'),
                        (4, 'knowledge', 'General knowledge that can be answered without web search')
                    ]
                    
                    cursor.executemany('''
                        INSERT INTO intent_categories (id, name, description)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (id) DO UPDATE SET 
                        name = EXCLUDED.name, 
                        description = EXCLUDED.description
                    ''', categories)
                    
                    conn.commit()
        except Exception as e:
            print(f"PostgreSQL initialization failed: {e}")
            raise
            
            # Create training data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS training_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    intent INTEGER NOT NULL,
                    intent_name TEXT NOT NULL,
                    confidence REAL DEFAULT 1.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create intent categories table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS intent_categories (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL
                )
            ''')
            
            # Insert intent categories
            categories = [
                (0, 'automation', 'Desktop automation tasks like opening apps, clicking buttons'),
                (1, 'conversation', 'General chat, greetings, philosophical questions'),
                (2, 'system_info', 'Local system information queries'),
                (3, 'web_search', 'Real-time information that needs web search'),
                (4, 'knowledge', 'General knowledge that can be answered without web search')
            ]
            
            cursor.executemany('''
                INSERT OR REPLACE INTO intent_categories (id, name, description)
                VALUES (?, ?, ?)
            ''', categories)
            
            conn.commit()
    
    def add_training_data(self, text: str, intent: int, intent_name: str, confidence: float = 1.0):
        """Add a single training example."""
        with psycopg2.connect(**self.db_config) as conn:
            with conn.cursor() as cursor:
                cursor.execute('''
                    INSERT INTO training_data (text, intent, intent_name, confidence)
                    VALUES (%s, %s, %s, %s)
                ''', (text, intent, intent_name, confidence))
                conn.commit()
    
    def add_bulk_training_data(self, data: List[Tuple[str, int, str]]):
        """Add multiple training examples at once."""
        with psycopg2.connect(**self.db_config) as conn:
            with conn.cursor() as cursor:
                cursor.executemany('''
                    INSERT INTO training_data (text, intent, intent_name)
                    VALUES (%s, %s, %s)
                ''', data)
                conn.commit()
    
    def get_training_data(self) -> List[Tuple[str, int]]:
        """Get all training data as (text, intent) tuples."""
        with psycopg2.connect(**self.db_config) as conn:
            with conn.cursor() as cursor:
                cursor.execute('SELECT text, intent FROM training_data ORDER BY intent, text')
                return cursor.fetchall()
    
    def get_training_data_by_intent(self, intent: int) -> List[str]:
        """Get all training texts for a specific intent."""
        with psycopg2.connect(**self.db_config) as conn:
            with conn.cursor() as cursor:
                cursor.execute('SELECT text FROM training_data WHERE intent = %s ORDER BY text', (intent,))
                return [row[0] for row in cursor.fetchall()]
    
    def get_intent_stats(self) -> Dict[str, int]:
        """Get statistics about training data per intent."""
        with psycopg2.connect(**self.db_config) as conn:
            with conn.cursor() as cursor:
                cursor.execute('''
                    SELECT intent_name, COUNT(*) 
                    FROM training_data 
                    GROUP BY intent, intent_name 
                    ORDER BY intent
                ''')
                return dict(cursor.fetchall())
    
    def populate_training_data(self):
        """Populate the database with comprehensive training data."""
        # Check if already populated
        with psycopg2.connect(**self.db_config) as conn:
            with conn.cursor() as cursor:
                cursor.execute('SELECT COUNT(*) FROM training_data')
                count = cursor.fetchone()[0]
        
        if count > 100:  # Already populated
            return
        
        # Comprehensive training data
        training_examples = [
            # AUTOMATION (0) - Desktop control tasks
            ("open chrome", 0, "automation"),
            ("open browser", 0, "automation"),
            ("launch edge", 0, "automation"),
            ("start firefox", 0, "automation"),
            ("open notepad", 0, "automation"),
            ("launch calculator", 0, "automation"),
            ("start spotify", 0, "automation"),
            ("open file explorer", 0, "automation"),
            ("launch terminal", 0, "automation"),
            ("open command prompt", 0, "automation"),
            ("click the button", 0, "automation"),
            ("press enter", 0, "automation"),
            ("type this text", 0, "automation"),
            ("scroll down", 0, "automation"),
            ("scroll up", 0, "automation"),
            ("close window", 0, "automation"),
            ("minimize app", 0, "automation"),
            ("maximize window", 0, "automation"),
            ("switch tabs", 0, "automation"),
            ("take screenshot", 0, "automation"),
            ("save file", 0, "automation"),
            ("copy text", 0, "automation"),
            ("paste clipboard", 0, "automation"),
            ("select all", 0, "automation"),
            ("drag and drop", 0, "automation"),
            ("right click", 0, "automation"),
            ("double click", 0, "automation"),
            ("open task manager", 0, "automation"),
            ("can you open chrome", 0, "automation"),
            ("please launch notepad", 0, "automation"),
            ("help me open calculator", 0, "automation"),
            ("start the browser for me", 0, "automation"),
            ("could you click that button", 0, "automation"),
            ("can you type this", 0, "automation"),
            ("please scroll down", 0, "automation"),
            ("help me close this", 0, "automation"),
            ("open settings", 0, "automation"),
            ("launch control panel", 0, "automation"),
            ("start task manager", 0, "automation"),
            
            # CONVERSATION (1) - Chat, greetings, casual talk
            ("hello", 1, "conversation"),
            ("hi", 1, "conversation"),
            ("hey", 1, "conversation"),
            ("good morning", 1, "conversation"),
            ("good evening", 1, "conversation"),
            ("how are you", 1, "conversation"),
            ("what's up", 1, "conversation"),
            ("how's it going", 1, "conversation"),
            ("nice to meet you", 1, "conversation"),
            ("what is your name", 1, "conversation"),
            ("tell me about yourself", 1, "conversation"),
            ("how do you work", 1, "conversation"),
            ("what can you do", 1, "conversation"),
            ("are you intelligent", 1, "conversation"),
            ("do you have feelings", 1, "conversation"),
            ("tell me a joke", 1, "conversation"),
            ("make me laugh", 1, "conversation"),
            ("you're funny", 1, "conversation"),
            ("thanks for helping", 1, "conversation"),
            ("you're awesome", 1, "conversation"),
            ("good job", 1, "conversation"),
            ("well done", 1, "conversation"),
            ("i appreciate it", 1, "conversation"),
            ("thank you", 1, "conversation"),
            ("goodbye", 1, "conversation"),
            ("see you later", 1, "conversation"),
            ("have a good day", 1, "conversation"),
            ("take care", 1, "conversation"),
            ("like why me always bro", 1, "conversation"),
            ("bro what's wrong", 1, "conversation"),
            ("dude help me out", 1, "conversation"),
            ("man this is crazy", 1, "conversation"),
            ("yo what's good", 1, "conversation"),
            ("sup friend", 1, "conversation"),
            ("hey buddy", 1, "conversation"),
            ("wassup mate", 1, "conversation"),
            
            # SYSTEM_INFO (2) - Local system queries
            ("show disk space", 2, "system_info"),
            ("check memory usage", 2, "system_info"),
            ("how much ram", 2, "system_info"),
            ("show cpu usage", 2, "system_info"),
            ("system information", 2, "system_info"),
            ("get hardware info", 2, "system_info"),
            ("show battery status", 2, "system_info"),
            ("check uptime", 2, "system_info"),
            ("system uptime", 2, "system_info"),
            ("get windows version", 2, "system_info"),
            ("show network info", 2, "system_info"),
            ("what's my ip address", 2, "system_info"),
            ("show running processes", 2, "system_info"),
            ("list installed programs", 2, "system_info"),
            ("show available drives", 2, "system_info"),
            ("check disk health", 2, "system_info"),
            ("get bios info", 2, "system_info"),
            ("show motherboard details", 2, "system_info"),
            ("check temperature", 2, "system_info"),
            ("show graphics card", 2, "system_info"),
            ("get processor info", 2, "system_info"),
            ("list files in downloads", 2, "system_info"),
            ("show desktop files", 2, "system_info"),
            ("check folder size", 2, "system_info"),
            ("get user accounts", 2, "system_info"),
            ("show startup programs", 2, "system_info"),
            ("check windows updates", 2, "system_info"),
            
            # WEB_SEARCH (3) - Real-time information needing web
            ("trending news today", 3, "web_search"),
            ("latest news", 3, "web_search"),
            ("breaking news", 3, "web_search"),
            ("current news", 3, "web_search"),
            ("what's happening today", 3, "web_search"),
            ("recent news updates", 3, "web_search"),
            ("trending topics", 3, "web_search"),
            ("viral news", 3, "web_search"),
            ("latest headlines", 3, "web_search"),
            ("weather today", 3, "web_search"),
            ("current weather", 3, "web_search"),
            ("weather forecast", 3, "web_search"),
            ("temperature now", 3, "web_search"),
            ("weather in london", 3, "web_search"),
            ("rain forecast", 3, "web_search"),
            ("stock price", 3, "web_search"),
            ("tesla stock", 3, "web_search"),
            ("bitcoin price", 3, "web_search"),
            ("market news", 3, "web_search"),
            ("crypto prices", 3, "web_search"),
            ("exchange rate", 3, "web_search"),
            ("live sports scores", 3, "web_search"),
            ("cricket score", 3, "web_search"),
            ("football results", 3, "web_search"),
            ("match results", 3, "web_search"),
            ("who won the game", 3, "web_search"),
            ("man of the match", 3, "web_search"),
            ("player of the series", 3, "web_search"),
            ("tournament results", 3, "web_search"),
            ("championship winner", 3, "web_search"),
            ("asia cup final", 3, "web_search"),
            ("trending songs", 3, "web_search"),
            ("viral videos", 3, "web_search"),
            ("youtube trending", 3, "web_search"),
            ("new movie releases", 3, "web_search"),
            ("box office results", 3, "web_search"),
            ("celebrity news", 3, "web_search"),
            ("latest music", 3, "web_search"),
            ("trending hashtags", 3, "web_search"),
            ("social media trends", 3, "web_search"),
            ("current events", 3, "web_search"),
            ("recent updates", 3, "web_search"),
            ("live updates", 3, "web_search"),
            ("real time data", 3, "web_search"),
            ("recent earthquake", 3, "web_search"),
            ("traffic updates", 3, "web_search"),
            ("flight status", 3, "web_search"),
            
            # KNOWLEDGE (4) - General knowledge (no web needed)
            ("what is python", 4, "knowledge"),
            ("explain machine learning", 4, "knowledge"),
            ("how does ai work", 4, "knowledge"),
            ("what is programming", 4, "knowledge"),
            ("define artificial intelligence", 4, "knowledge"),
            ("what is the capital of france", 4, "knowledge"),
            ("how tall is mount everest", 4, "knowledge"),
            ("who invented the computer", 4, "knowledge"),
            ("what is quantum physics", 4, "knowledge"),
            ("explain photosynthesis", 4, "knowledge"),
            ("what is dna", 4, "knowledge"),
            ("how do computers work", 4, "knowledge"),
            ("what is the internet", 4, "knowledge"),
            ("explain gravity", 4, "knowledge"),
            ("what is relativity", 4, "knowledge"),
            ("how does gps work", 4, "knowledge"),
            ("what is blockchain", 4, "knowledge"),
            ("explain algorithms", 4, "knowledge"),
            ("what is database", 4, "knowledge"),
            ("how does wifi work", 4, "knowledge"),
            ("what is http", 4, "knowledge"),
            ("explain tcp ip", 4, "knowledge"),
            ("what is operating system", 4, "knowledge"),
            ("how does cpu work", 4, "knowledge"),
            ("what is ram", 4, "knowledge"),
            ("explain hard drive", 4, "knowledge"),
            ("what is software", 4, "knowledge"),
            ("define hardware", 4, "knowledge"),
            ("what is cloud computing", 4, "knowledge"),
            ("explain virtual reality", 4, "knowledge"),
            ("what is augmented reality", 4, "knowledge"),
            ("how does camera work", 4, "knowledge"),
            ("what is semiconductor", 4, "knowledge"),
            ("explain neural networks", 4, "knowledge"),
            ("what is deep learning", 4, "knowledge"),
            ("how does internet work", 4, "knowledge"),
            ("what is sql", 4, "knowledge"),
            ("explain apis", 4, "knowledge"),
            ("what is json", 4, "knowledge"),
            ("define xml", 4, "knowledge"),
        ]
        
        # Add all training data
        self.add_bulk_training_data(training_examples)
        print(f"Added {len(training_examples)} training examples to database")

if __name__ == "__main__":
    # PostgreSQL configuration
    postgresql_config = get_default_postgresql_config()
    
    # Test the database - PostgreSQL only
    db = IntentTrainingDatabase(postgresql_config)
    
    stats = db.get_intent_stats()
    print("Intent training database statistics:")
    for intent, count in stats.items():
        print(f"  {intent}: {count} examples")