#!/usr/bin/env python3
"""
Universal Assistant - Smart Friend-like Data Retrieval System

This system can handle ANY question just like a smart friend:
- System information and commands  
- Conversational context awareness
- Smart query enhancement and context understanding

Author: Truvo Assistant
Version: 4.0 - Clean Universal Friend (No Web Scraping)
"""

import requests
import subprocess
import json
import re
import time
import os
import platform
import socket
import urllib.parse
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
from urllib.parse import quote, urljoin

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    from bs4 import BeautifulSoup
    HAS_BEAUTIFULSOUP = True
except ImportError:
    HAS_BEAUTIFULSOUP = False


class UniversalAssistant:
    """Universal Assistant - Can answer ANY question like a smart friend."""
    
    def __init__(self):
        """Initialize the Universal Assistant."""
        self.system_info = self._get_system_info()
        self.os_name = platform.system()
        self.conversation_context = []
        print("✅ Universal Assistant loaded - Ready to answer ANY question!")
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Gather comprehensive system information."""
        info = {
            'hostname': socket.gethostname(),
            'os': platform.system(),
            'os_version': platform.version(),
            'os_release': platform.release(),
            'architecture': platform.architecture()[0],
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'username': os.getenv('USERNAME', os.getenv('USER', 'unknown')),
            'user_profile': os.path.expanduser('~')
        }
        
        if HAS_PSUTIL:
            info['cpu_count'] = psutil.cpu_count()
            info['boot_time'] = psutil.boot_time()
        
        return info
    
    def answer_anything(self, query: str) -> Dict[str, Any]:
        """
        Universal method to answer ANY question - the main interface.
        Like talking to a smart friend who knows everything about your system.
        """
        result = {
            'timestamp': datetime.now().isoformat(),
            'original_query': query,
            'success': False
        }
        
        try:
            # Step 1: Enhance query with conversational context
            enhanced_query = self._enhance_with_context(query)
            if enhanced_query != query:
                result['context_used'] = True
                result['enhanced_query'] = enhanced_query
            
            # Step 2: Determine query type and approach
            query_type = self._classify_query(enhanced_query)
            result['query_type'] = query_type
            
            # Step 3: Handle based on query type
            if query_type == 'system':
                result.update(self._handle_system_query(enhanced_query))
            elif query_type == 'command':
                result.update(self._handle_command_query(enhanced_query))
            elif query_type == 'live_data':
                result.update(self._handle_live_data_query(enhanced_query))
            elif query_type == 'factual':
                result.update(self._handle_factual_query(enhanced_query))
            elif query_type == 'conversational':
                result.update(self._handle_conversational_query(enhanced_query))
            else:
                # Default: provide helpful message instead of web search
                result.update({
                    'success': True,
                    'answer': f"I understand you're asking about '{query}'. Currently, web search is disabled. I can help with system information, commands, and general conversation. What specific information do you need?",
                    'data': {'type': 'web_search_disabled'}
                })
            
            # Step 4: If primary method failed, provide fallback response instead of web search
            if not result.get('success') and query_type != 'web_search':
                result.update({
                    'success': True,
                    'answer': f"I couldn't process '{query}' with the current methods. Web search is currently disabled. Is there something specific about your system or a command you'd like me to help with?",
                    'data': {'type': 'fallback_without_web_search'},
                    'fallback_used': True
                })
                    
        except Exception as e:
            result['error'] = str(e)
            # Provide helpful error message without web search
            result.update({
                'success': True,
                'answer': f"I encountered an error processing '{query}': {str(e)}. I can help with system information, commands, and conversation. What would you like to know?",
                'data': {'type': 'error_fallback_without_web_search'},
                'error_recovery': True
            })
        
        return result
    
    def _enhance_with_context(self, query: str) -> str:
        """Enhance query with conversational context."""
        if not self.conversation_context:
            return query
        
        query_lower = query.lower()
        
        # Context enhancement patterns
        context_entities = []
        
        # Look for entities in recent conversation
        for exchange in reversed(self.conversation_context[-5:]):  # Last 5 exchanges
            if exchange.get('role') == 'user':
                user_msg = exchange.get('content', '').lower()
                
                # Extract potential entities (simple approach)
                entities = self._extract_entities(user_msg)
                context_entities.extend(entities)
        
        # Remove duplicates while preserving order
        context_entities = list(dict.fromkeys(context_entities))
        
        # If current query is vague and we have context, enhance it
        if len(query.split()) <= 3 and context_entities:
            if any(word in query_lower for word in ['it', 'that', 'this', 'more', 'details']):
                # Use the most recent entity
                return f"{query} {context_entities[0]}"
        
        return query
    
    def _extract_entities(self, text: str) -> List[str]:
        """Simple entity extraction from text."""
        entities = []
        
        # Look for quoted strings
        quoted = re.findall(r'"([^"]+)"', text)
        entities.extend(quoted)
        
        # Look for capitalized words (potential proper nouns)
        capitalized = re.findall(r'\\b[A-Z][a-zA-Z]+\\b', text)
        entities.extend(capitalized)
        
        # Look for technical terms
        tech_terms = re.findall(r'\\b(?:API|URL|HTTP|JSON|XML|database|server|application)\\b', text, re.IGNORECASE)
        entities.extend(tech_terms)
        
        return entities[:3]  # Limit to top 3
    
    def _classify_query(self, query: str) -> str:
        """Classify the type of query to determine best approach."""
        query_lower = query.lower()
        
        # System information queries (HIGHEST PRIORITY)
        system_keywords = [
            'system', 'computer', 'pc', 'this machine', 'my computer', 'hardware', 'software', 'running on',
            'memory', 'ram', 'free memory', 'available memory', 'disk space', 'storage', 'cpu', 'processor',
            'username', 'user', 'hostname', 'uptime', 'os', 'operating system', 'windows', 'version',
            'name of this pc', 'computer name', 'pc name', 'machine name', 'system name', 'this pc'
        ]
        if any(word in query_lower for word in system_keywords):
            return 'system'
        
        # Command queries (SECOND PRIORITY)
        command_patterns = [
            'how many files', 'list files', 'show files', 'directory', 'folder content', 'files in',
            'count files', 'number of files', 'file count', 'disk usage', 'folder size', 'directory size'
        ]
        if any(pattern in query_lower for pattern in command_patterns):
            return 'command'
            
        # Conversational queries (THIRD PRIORITY - before web search!)
        conversational_keywords = [
            'thank you', 'thanks', 'hello', 'hi', 'bye', 'goodbye', 'how are you',
            'good morning', 'good evening', 'good night', 'what', 'hmm', 'ok', 'okay',
            'cool', 'nice', 'great', 'awesome', 'perfect', 'yes', 'no', 'maybe'
        ]
        if any(word in query_lower for word in conversational_keywords):
            return 'conversational'
        
        # Short queries that are likely conversational
        if len(query.strip()) <= 3:  # Single words like "what", "ok", "hi"
            return 'conversational'
        
        # Live data queries (FOURTH PRIORITY)
        live_data_keywords = [
            'live', 'current', 'latest', 'now', 'today', 'score', 'price', 'weather', 'stock', 'news',
            'breaking', 'update', 'real time', 'fresh', 'recent', 'happening'
        ]
        if any(word in query_lower for word in live_data_keywords):
            return 'live_data'
        
        # Factual queries (FIFTH PRIORITY)
        factual_patterns = [
            'what is', 'who is', 'when is', 'where is', 'how to', 'definition', 'explain',
            'tell me about', 'information about', 'details about', 'describe'
        ]
        if any(pattern in query_lower for pattern in factual_patterns):
            return 'factual'
        
        # Default to conversational for short unclear queries, web search for complex ones
        if len(query.strip()) < 20:
            return 'conversational'
        else:
            return 'web_search'
    
    def _handle_system_query(self, query: str) -> Dict[str, Any]:
        """Handle system-related queries."""
        query_lower = query.lower()
        
        # Get relevant system information
        relevant_info = {}
        
        if any(word in query_lower for word in ['cpu', 'processor', 'cores']):
            if HAS_PSUTIL:
                relevant_info['cpu'] = {
                    'name': self.system_info.get('processor', 'Unknown'),
                    'cores': self.system_info.get('cpu_count', 'Unknown'),
                    'usage': f"{psutil.cpu_percent(interval=1):.1f}%"
                }
        
        if any(word in query_lower for word in ['memory', 'ram', 'storage']):
            if HAS_PSUTIL:
                mem = psutil.virtual_memory()
                relevant_info['memory'] = {
                    'total': f"{mem.total / (1024**3):.1f} GB",
                    'available': f"{mem.available / (1024**3):.1f} GB",
                    'usage': f"{mem.percent}%"
                }
        
        if any(word in query_lower for word in ['disk', 'storage', 'space']):
            if HAS_PSUTIL:
                disk = psutil.disk_usage('C:\\\\' if self.os_name == 'Windows' else '/')
                relevant_info['disk'] = {
                    'total': f"{disk.total / (1024**3):.1f} GB",
                    'free': f"{disk.free / (1024**3):.1f} GB",
                    'usage': f"{(disk.used / disk.total) * 100:.1f}%"
                }
        
        if any(word in query_lower for word in ['os', 'operating system', 'windows', 'version']):
            relevant_info['os'] = {
                'name': self.system_info['os'],
                'version': self.system_info['os_version'],
                'release': self.system_info['os_release'],
                'architecture': self.system_info['architecture']
            }
        
        # If no specific system info requested, provide overview
        if not relevant_info:
            relevant_info = {
                'overview': {
                    'hostname': self.system_info['hostname'],
                    'os': f"{self.system_info['os']} {self.system_info['os_release']}",
                    'user': self.system_info['username'],
                    'uptime': self._get_uptime()
                }
            }
        
        return {
            'success': True,
            'answer': self._format_system_answer(relevant_info, query),
            'data': {'system_info': relevant_info, 'type': 'system_query'}
        }
    
    def _handle_command_query(self, query: str) -> Dict[str, Any]:
        """Handle command-related queries by executing system commands."""
        try:
            command = self._translate_to_command(query)
            if not command:
                return {
                    'success': False,
                    'error': 'Could not translate query to system command'
                }
            
            # Execute command
            result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
            output = result.stdout if result.returncode == 0 else result.stderr
            
            if result.returncode == 0:
                return {
                    'success': True,
                    'answer': self._format_command_answer(output, query),
                    'data': {
                        'command': command,
                        'output': output,
                        'type': 'command_execution'
                    }
                }
            else:
                return {
                    'success': False,
                    'error': f"Command failed: {output}",
                    'data': {'command': command, 'return_code': result.returncode}
                }
        
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': 'Command timed out after 30 seconds'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Command execution failed: {str(e)}'
            }
    
    def _handle_live_data_query(self, query: str) -> Dict[str, Any]:
        """Handle live data queries - currently disabled."""
        return {
            'success': True,
            'answer': f"Live data queries like '{query}' are temporarily disabled while we improve the web search system. I can help with system information, commands, and general questions instead!",
            'data': {'type': 'live_data_disabled'}
        }
    
    def _handle_factual_query(self, query: str) -> Dict[str, Any]:
        """Handle factual queries - currently disabled."""  
        return {
            'success': True,
            'answer': f"Factual queries like '{query}' are temporarily disabled while we improve the web search system. I can help with system information, commands, and conversation instead!",
            'data': {'type': 'factual_disabled'}
        }
    
    def _handle_conversational_query(self, query: str) -> Dict[str, Any]:
        """Handle conversational queries."""
        query_lower = query.lower().strip()
        
        # Predefined conversational responses
        responses = {
            'hello': "Hello! I'm Truvo, your universal assistant. I can help you with system info, commands, and general conversation!",
            'hi': "Hi there! What can I help you with today?",
            'how are you': "I'm doing great! Ready to help you with system info, commands, or just chat.",
            'thank you': "You're welcome! Always happy to help with anything you need.",
            'thanks': "No problem! Feel free to ask me anything else.",
            'bye': "Goodbye! It was great chatting with you.",
            'goodbye': "See you later! I'm always here when you need help.",
            'what': "What would you like to know? I can help with system info, commands, or just chat!",
            'ok': "Great! What else can I help you with?",
            'okay': "Perfect! Is there anything else you'd like to know?",
            'cool': "Glad you think so! Anything else I can help with?",
            'nice': "Thank you! What else can I assist you with?",
            'hmm': "Is there something specific you'd like to know more about?",
            'yes': "Excellent! What would you like to do next?",
            'no': "No problem! Let me know if you need anything else."
        }
        
        # Check for exact matches first
        if query_lower in responses:
            return {
                'success': True,
                'answer': responses[query_lower],
                'data': {'type': 'conversational_response'}
            }
        
        # Check for partial matches
        for trigger, response in responses.items():
            if trigger in query_lower:
                return {
                    'success': True,
                    'answer': response,
                    'data': {'type': 'conversational_response'}
                }
        
        # For other short conversational queries, provide a general helpful response
        if len(query_lower) < 20:
            return {
                'success': True,
                'answer': f"I understand you said '{query}'. I'm here to help with anything you need - system information, commands, or general questions. What would you like to know?",
                'data': {'type': 'general_conversational_response'}
            }
        
        # For longer queries that reached here, they might actually need web search
        return {
            'success': False,
            'reason': 'conversational_query_needs_escalation'
        }
    
    def _translate_to_command(self, query: str) -> str:
        """Translate natural language to system commands."""
        # Common folder paths
        user_profile = self.system_info['user_profile']
        common_folders = {
            'downloads': os.path.join(user_profile, 'Downloads'),
            'documents': os.path.join(user_profile, 'Documents'), 
            'desktop': os.path.join(user_profile, 'Desktop'),
            'pictures': os.path.join(user_profile, 'Pictures'),
            'music': os.path.join(user_profile, 'Music'),
            'videos': os.path.join(user_profile, 'Videos')
        }
        
        query_lower = query.lower()
        
        # File counting queries
        if 'files in' in query_lower:
            for folder_name, folder_path in common_folders.items():
                if folder_name in query_lower:
                    if self.os_name == 'Windows':
                        return f'powershell "Get-ChildItem \'{folder_path}\' | Format-Table Name, Length, LastWriteTime"'
                    else:
                        return f'ls -la "{folder_path}"'
        
        # Directory listing
        if any(pattern in query_lower for pattern in ['list files', 'show files']):
            if self.os_name == 'Windows':
                return 'dir'
            else:
                return 'ls -la'
        
        # File count in current directory
        if any(pattern in query_lower for pattern in ['how many files', 'count files']):
            if self.os_name == 'Windows':
                return 'powershell "(Get-ChildItem).Count"'
            else:
                return 'ls -1 | wc -l'
        
        return None
    
    def _format_system_answer(self, info: Dict, query: str) -> str:
        """Format system information into a readable answer."""
        if 'memory' in info:
            mem = info['memory']
            return f"Memory: {mem['available']} available out of {mem['total']} total ({mem['usage']} used)"
        
        elif 'disk' in info:
            disk = info['disk']
            return f"Disk: {disk['free']} free out of {disk['total']} total ({disk['usage']} used)"
        
        elif 'cpu' in info:
            cpu = info['cpu']
            return f"CPU: {cpu['name']} with {cpu['cores']} cores (Current usage: {cpu['usage']})"
        
        elif 'os' in info:
            os_info = info['os']
            return f"Operating System: {os_info['name']} {os_info['release']} ({os_info['architecture']})"
        
        elif 'overview' in info:
            overview = info['overview']
            return f"Your system: {overview['hostname']} running {overview['os']}, logged in as {overview['user']}. {overview['uptime']}"
        
        return "System information retrieved successfully."
    
    def _format_command_answer(self, output: str, query: str) -> str:
        """Format command output into a readable answer."""
        if not output.strip():
            return "Command executed successfully with no output."
        
        # Truncate very long outputs
        if len(output) > 2000:
            return output[:2000] + "\\n\\n... (output truncated)"
        
        return output.strip()
    
    def _get_uptime(self) -> str:
        """Get system uptime in a readable format."""
        if not HAS_PSUTIL:
            return "Uptime information not available"
        
        boot_time = psutil.boot_time()
        uptime_seconds = time.time() - boot_time
        
        days = int(uptime_seconds // 86400)
        hours = int((uptime_seconds % 86400) // 3600)
        minutes = int((uptime_seconds % 3600) // 60)
        
        if days > 0:
            return f"Uptime: {days} days, {hours} hours"
        elif hours > 0:
            return f"Uptime: {hours} hours, {minutes} minutes"
        else:
            return f"Uptime: {minutes} minutes"


# Create global instance for compatibility
universal_assistant = UniversalAssistant()

# Main API functions
def retrieve_data(query: str) -> str:
    """Main function to retrieve any data - universal entry point."""
    result = universal_assistant.answer_anything(query)
    return result.get('answer', 'No answer available')

def answer_anything(query: str) -> str:
    """Alias for retrieve_data."""
    return retrieve_data(query)

def get_system_info() -> Dict[str, Any]:
    """Get system information."""
    return universal_assistant.system_info

# Legacy compatibility
data_retriever = universal_assistant