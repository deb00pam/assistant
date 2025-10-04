"""
Enhanced Storage System for Truvo - Context Memory & Data Persistence

Handles:
- Conversation history & context
- GUI automation patterns & learning
- User preferences & settings
- Application knowledge base
- Task execution history
"""

import json
import os
import sqlite3
import pickle
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

# File paths
USER_DATA_FILE = "user_data.json"
CONTEXT_DB = "context_memory.db"
PATTERNS_FILE = "gui_patterns.pkl"
CONVERSATIONS_DIR = "conversations"

@dataclass
class ConversationTurn:
    """Single conversation turn with timestamp"""
    timestamp: str
    user_message: str
    assistant_response: str
    context_tags: List[str]
    importance_score: float

@dataclass
class GUIPattern:
    """Learned GUI automation pattern"""
    app_name: str
    task_description: str
    ui_elements: List[Dict]
    action_sequence: List[Dict]
    success_rate: float
    last_used: str
    usage_count: int

@dataclass
class TaskMemory:
    """Memory of a completed task"""
    goal: str
    steps_taken: List[Dict]
    success: bool
    duration: float
    screenshots: List[str]
    lessons_learned: str

class ContextMemoryManager:
    """Advanced context and memory management system"""
    
    def __init__(self, max_context_turns: int = 50):
        self.max_context_turns = max_context_turns
        self.conversations_dir = Path(CONVERSATIONS_DIR)
        self.conversations_dir.mkdir(exist_ok=True)
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database for structured memory"""
        self.conn = sqlite3.connect(CONTEXT_DB)
        cursor = self.conn.cursor()
        
        # Conversation history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                user_message TEXT,
                assistant_response TEXT,
                context_tags TEXT,
                importance_score REAL,
                session_id TEXT
            )
        ''')
        
        # GUI patterns table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS gui_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                app_name TEXT,
                task_description TEXT,
                pattern_data TEXT,
                success_rate REAL,
                usage_count INTEGER,
                last_used TEXT
            )
        ''')
        
        # Task execution history
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS task_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                goal TEXT,
                execution_data TEXT,
                success BOOLEAN,
                duration REAL,
                timestamp TEXT
            )
        ''')
        
        self.conn.commit()
    
    def save_conversation_turn(self, user_msg: str, assistant_response: str, 
                             context_tags: List[str] = None, importance: float = 1.0,
                             session_id: str = None):
        """Save a conversation turn with context"""
        cursor = self.conn.cursor()
        timestamp = datetime.now().isoformat()
        context_tags = context_tags or []
        session_id = session_id or self.get_current_session_id()
        
        cursor.execute('''
            INSERT INTO conversations 
            (timestamp, user_message, assistant_response, context_tags, importance_score, session_id)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (timestamp, user_msg, assistant_response, json.dumps(context_tags), importance, session_id))
        
        self.conn.commit()
    
    def get_relevant_context(self, current_query: str, limit: int = 10) -> List[ConversationTurn]:
        """Get relevant conversation history with advanced semantic matching"""
        cursor = self.conn.cursor()
        
        # Get all recent conversations for semantic analysis
        cursor.execute('''
            SELECT timestamp, user_message, assistant_response, context_tags, importance_score
            FROM conversations 
            ORDER BY timestamp DESC
            LIMIT 100
        ''')
        
        all_conversations = [ConversationTurn(*row) for row in cursor.fetchall()]
        
        # Calculate relevance scores
        scored_conversations = []
        query_words = set(current_query.lower().split())
        
        for conv in all_conversations:
            score = self._calculate_relevance_score(conv, query_words, current_query)
            if score > 0.1:  # Minimum relevance threshold
                scored_conversations.append((score, conv))
        
        # Sort by relevance score and return top results
        scored_conversations.sort(key=lambda x: x[0], reverse=True)
        return [conv for _, conv in scored_conversations[:limit]]
    
    def _calculate_relevance_score(self, conv: ConversationTurn, query_words: set, original_query: str) -> float:
        """Calculate semantic relevance score for a conversation"""
        score = 0.0
        
        # Text content matching
        conv_text = (conv.user_message + " " + conv.assistant_response).lower()
        conv_words = set(conv_text.split())
        
        # Word overlap score
        word_overlap = len(query_words.intersection(conv_words)) / len(query_words)
        score += word_overlap * 0.4
        
        # Exact phrase matching
        if original_query.lower() in conv_text:
            score += 0.3
        
        # Context tags matching
        if conv.context_tags:
            try:
                tags = json.loads(conv.context_tags) if isinstance(conv.context_tags, str) else conv.context_tags
                for word in query_words:
                    if any(word in tag.lower() for tag in tags):
                        score += 0.2
            except:
                pass
        
        # Importance boost
        score += conv.importance_score * 0.1
        
        # Recency boost (conversations from last 24 hours get bonus)
        try:
            conv_time = datetime.fromisoformat(conv.timestamp)
            hours_ago = (datetime.now() - conv_time).total_seconds() / 3600
            if hours_ago < 24:
                score += 0.1 * (1 - hours_ago / 24)
        except:
            pass
        
        return min(score, 1.0)  # Cap at 1.0
    
    def save_gui_pattern(self, pattern: GUIPattern):
        """Save a learned GUI automation pattern"""
        cursor = self.conn.cursor()
        pattern_data = json.dumps(asdict(pattern))
        
        cursor.execute('''
            INSERT OR REPLACE INTO gui_patterns
            (app_name, task_description, pattern_data, success_rate, usage_count, last_used)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (pattern.app_name, pattern.task_description, pattern_data, 
              pattern.success_rate, pattern.usage_count, pattern.last_used))
        
        self.conn.commit()
    
    def get_gui_patterns(self, app_name: str = None, task_type: str = None) -> List[GUIPattern]:
        """Retrieve GUI patterns for specific app or task"""
        cursor = self.conn.cursor()
        
        if app_name and task_type:
            cursor.execute('''
                SELECT pattern_data FROM gui_patterns 
                WHERE app_name = ? AND task_description LIKE ?
                ORDER BY success_rate DESC, usage_count DESC
            ''', (app_name, f'%{task_type}%'))
        elif app_name:
            cursor.execute('''
                SELECT pattern_data FROM gui_patterns 
                WHERE app_name = ?
                ORDER BY success_rate DESC, usage_count DESC
            ''', (app_name,))
        else:
            cursor.execute('''
                SELECT pattern_data FROM gui_patterns 
                ORDER BY success_rate DESC, usage_count DESC
            ''')
        
        patterns = []
        for (pattern_data,) in cursor.fetchall():
            data = json.loads(pattern_data)
            patterns.append(GUIPattern(**data))
        
        return patterns
    
    def save_task_execution(self, task_memory: TaskMemory):
        """Save task execution for learning"""
        cursor = self.conn.cursor()
        execution_data = json.dumps(asdict(task_memory))
        timestamp = datetime.now().isoformat()
        
        cursor.execute('''
            INSERT INTO task_history
            (goal, execution_data, success, duration, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (task_memory.goal, execution_data, task_memory.success, 
              task_memory.duration, timestamp))
        
        self.conn.commit()
    
    def get_current_session_id(self) -> str:
        """Generate/get current session ID"""
        today = datetime.now().strftime("%Y%m%d")
        return f"session_{today}_{datetime.now().hour}"
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old conversation data to manage storage"""
        cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).isoformat()
        cursor = self.conn.cursor()
        
        cursor.execute('DELETE FROM conversations WHERE timestamp < ?', (cutoff_date,))
        cursor.execute('DELETE FROM task_history WHERE timestamp < ?', (cutoff_date,))
        
        self.conn.commit()
    
    def learn_user_preferences(self) -> Dict[str, Any]:
        """Analyze conversation patterns to learn user preferences"""
        cursor = self.conn.cursor()
        
        # Get recent conversations for pattern analysis
        cursor.execute('''
            SELECT user_message, assistant_response, context_tags, importance_score
            FROM conversations 
            WHERE timestamp > datetime('now', '-30 days')
            ORDER BY timestamp DESC
        ''')
        
        conversations = cursor.fetchall()
        preferences = {
            'frequent_topics': {},
            'preferred_response_style': 'detailed',  # vs 'concise'
            'common_tasks': {},
            'interaction_patterns': {},
            'time_preferences': {}
        }
        
        # Analyze frequent topics from context tags
        for _, _, tags_str, importance in conversations:
            if tags_str:
                try:
                    tags = json.loads(tags_str)
                    for tag in tags:
                        preferences['frequent_topics'][tag] = preferences['frequent_topics'].get(tag, 0) + importance
                except:
                    pass
        
        # Analyze common task patterns
        for user_msg, _, _, importance in conversations:
            # Extract potential tasks (messages starting with action words)
            action_words = ['open', 'create', 'find', 'search', 'help', 'explain', 'show']
            for word in action_words:
                if user_msg.lower().startswith(word):
                    preferences['common_tasks'][word] = preferences['common_tasks'].get(word, 0) + importance
        
        return preferences
    
    def get_contextual_suggestions(self, current_context: str) -> List[str]:
        """Generate smart suggestions based on conversation history"""
        preferences = self.learn_user_preferences()
        suggestions = []
        
        # Suggest based on frequent topics
        for topic, frequency in sorted(preferences['frequent_topics'].items(), 
                                     key=lambda x: x[1], reverse=True)[:3]:
            suggestions.append(f"Would you like help with {topic}?")
        
        # Suggest based on common tasks
        for task, frequency in sorted(preferences['common_tasks'].items(), 
                                    key=lambda x: x[1], reverse=True)[:2]:
            suggestions.append(f"I can help you {task} something")
        
        return suggestions[:5]  # Return top 5 suggestions
    
    def analyze_conversation_quality(self) -> Dict[str, float]:
        """Analyze conversation quality metrics"""
        cursor = self.conn.cursor()
        
        # Get conversation metrics
        cursor.execute('''
            SELECT 
                AVG(importance_score) as avg_importance,
                COUNT(*) as total_conversations,
                COUNT(DISTINCT DATE(timestamp)) as active_days
            FROM conversations 
            WHERE timestamp > datetime('now', '-30 days')
        ''')
        
        result = cursor.fetchone()
        avg_importance, total_convs, active_days = result
        
        metrics = {
            'average_importance': avg_importance or 0.0,
            'conversations_per_day': (total_convs / max(active_days, 1)) if active_days else 0.0,
            'engagement_score': min((avg_importance or 0) * (total_convs / 100), 1.0)
        }
        
        return metrics
    
    def optimize_memory_usage(self):
        """Optimize memory by compressing old conversations"""
        cursor = self.conn.cursor()
        
        # Compress conversations older than 7 days but keep high-importance ones
        week_ago = (datetime.now() - timedelta(days=7)).isoformat()
        
        cursor.execute('''
            SELECT id, user_message, assistant_response, importance_score
            FROM conversations 
            WHERE timestamp < ? AND importance_score < 0.7
        ''', (week_ago,))
        
        old_conversations = cursor.fetchall()
        
        # Create summaries for groups of related conversations
        summaries = {}
        for conv_id, user_msg, assistant_resp, importance in old_conversations:
            # Group by first word of user message (rough topic grouping)
            topic = user_msg.split()[0].lower() if user_msg else 'general'
            if topic not in summaries:
                summaries[topic] = {
                    'count': 0,
                    'sample_messages': [],
                    'avg_importance': 0
                }
            
            summaries[topic]['count'] += 1
            if len(summaries[topic]['sample_messages']) < 3:
                summaries[topic]['sample_messages'].append(user_msg[:100])
            summaries[topic]['avg_importance'] += importance
        
        # Save summaries and delete old conversations
        for topic, summary in summaries.items():
            if summary['count'] > 5:  # Only summarize if there are enough conversations
                summary['avg_importance'] /= summary['count']
                summary_text = f"Summarized {summary['count']} conversations about {topic}"
                
                cursor.execute('''
                    INSERT INTO conversations 
                    (timestamp, user_message, assistant_response, context_tags, importance_score, session_id)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (datetime.now().isoformat(), f"[SUMMARY] {topic}", summary_text, 
                      json.dumps([topic, 'summary']), summary['avg_importance'], 'summary'))
        
        # Delete the old conversations that were summarized
        cursor.execute('''
            DELETE FROM conversations 
            WHERE timestamp < ? AND importance_score < 0.7
        ''', (week_ago,))
        
        self.conn.commit()

# Legacy functions for backwards compatibility
def load_user_data() -> Dict[str, Any]:
    if os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_user_data(data: Dict[str, Any]):
    with open(USER_DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# Global instance
context_manager = ContextMemoryManager()
