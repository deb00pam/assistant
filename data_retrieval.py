#!/usr/bin/env python3
"""
Data Retrieval Module for Truvo Desktop Assistant

This module provides comprehensive data retrieval capabilities including:
- Web scraping with BeautifulSoup and requests
- Command execution with security and timeout handling
- API integrations for weather, news, and other real-time data
- Data parsing and formatting for AI consumption

Author: Truvo Assistant
Version: 1.0
"""

import requests
import subprocess
import json
import re
import time
import os
import tempfile
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from urllib.parse import urljoin, urlparse
from pathlib import Path

try:
    from bs4 import BeautifulSoup
    HAS_BEAUTIFULSOUP = True
except ImportError:
    HAS_BEAUTIFULSOUP = False
    print("Warning: BeautifulSoup not available. Web scraping will be limited.")

try:
    import feedparser
    HAS_FEEDPARSER = True
except ImportError:
    HAS_FEEDPARSER = False


class DataRetriever:
    """Main class for handling all data retrieval operations."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        self.command_timeout = 30  # seconds
        self.web_timeout = 10  # seconds
        self.cache = {}  # Simple in-memory cache
        self.cache_expiry = {}
        
    def retrieve_data(self, query: str, method: str = "auto") -> Dict[str, Any]:
        """
        Main method to retrieve data based on query and method.
        
        Args:
            query: The data to retrieve (URL, command, search term, etc.)
            method: "web", "command", "api", or "auto" to auto-detect
            
        Returns:
            Dictionary with retrieved data and metadata
        """
        result = {
            'success': False,
            'data': None,
            'method_used': method,
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'error': None,
            'metadata': {}
        }
        
        try:
            # Auto-detect method if not specified
            if method == "auto":
                method = self._detect_retrieval_method(query)
                result['method_used'] = method
            
            # Route to appropriate handler
            if method == "web":
                result.update(self._retrieve_web_data(query))
            elif method == "command":
                result.update(self._execute_command(query))
            elif method == "api":
                result.update(self._retrieve_api_data(query))
            elif method == "search":
                result.update(self._search_web(query))
            else:
                result['error'] = f"Unknown retrieval method: {method}"
            
            # Universal browser fallback check
            if result.get('success', False):
                satisfactory = self._is_result_satisfactory(result, query)
                if not satisfactory:
                    result['browser_fallback'] = True
                    result['search_query'] = query
                    result['fallback_reason'] = 'Limited or no relevant data found'
            else:
                # Failed results definitely need browser fallback
                result['browser_fallback'] = True
                result['search_query'] = query
                result['fallback_reason'] = result.get('error', 'Data retrieval failed')
                
        except Exception as e:
            result['error'] = str(e)
            result['success'] = False
            result['browser_fallback'] = True
            result['search_query'] = query
            result['fallback_reason'] = f'System error: {str(e)}'
            
        return result
    
    def _detect_retrieval_method(self, query: str) -> str:
        """Auto-detect the best retrieval method for the query."""
        query_lower = query.lower().strip()
        
        # Check for URLs
        if query_lower.startswith(('http://', 'https://', 'www.')):
            return "web"
        
        # Check for command patterns
        command_patterns = [
            r'^(dir|ls|pwd|cd|cat|type|more|ping|ipconfig|systeminfo|tasklist)',
            r'^(python|node|java|git|npm|pip)\s+',
            r'^[a-zA-Z]+\.exe\s*',
            r'^powershell\s+',
            r'^cmd\s+'
        ]
        
        for pattern in command_patterns:
            if re.search(pattern, query_lower):
                return "command"
        
        # Check for API-style queries
        api_patterns = [
            r'weather\s+in\s+',
            r'news\s+(about|on)\s+',
            r'stock\s+price\s+',
            r'exchange\s+rate\s+',
            r'translate\s+.+\s+to\s+'
        ]
        
        for pattern in api_patterns:
            if re.search(pattern, query_lower):
                return "api"
        
        # Default to web search
        return "search"
    
    def _retrieve_web_data(self, url: str) -> Dict[str, Any]:
        """Retrieve and parse data from a web URL."""
        if not HAS_BEAUTIFULSOUP:
            return {
                'success': False,
                'error': 'BeautifulSoup not available for web scraping'
            }
        
        try:
            # Normalize URL
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            # Check cache first
            cache_key = f"web_{url}"
            if self._is_cached(cache_key):
                cached_data = self.cache[cache_key]
                cached_data['from_cache'] = True
                return cached_data
            
            response = self.session.get(url, timeout=self.web_timeout)
            response.raise_for_status()
            
            # Parse content
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract useful information
            data = {
                'success': True,
                'url': url,
                'title': self._extract_title(soup),
                'text_content': self._extract_text_content(soup),
                'links': self._extract_links(soup, url),
                'images': self._extract_images(soup, url),
                'metadata': {
                    'status_code': response.status_code,
                    'content_type': response.headers.get('Content-Type', ''),
                    'content_length': len(response.content),
                    'encoding': response.encoding
                }
            }
            
            # Cache the result
            self._cache_data(cache_key, data, minutes=10)
            
            return data
            
        except requests.exceptions.Timeout:
            return {'success': False, 'error': f'Request timeout after {self.web_timeout} seconds'}
        except requests.exceptions.RequestException as e:
            return {'success': False, 'error': f'Request failed: {str(e)}'}
        except Exception as e:
            return {'success': False, 'error': f'Web scraping error: {str(e)}'}
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title."""
        title_tag = soup.find('title')
        if title_tag:
            return title_tag.get_text().strip()
        
        # Try h1 as fallback
        h1_tag = soup.find('h1')
        if h1_tag:
            return h1_tag.get_text().strip()
        
        return "No title found"
    
    def _extract_text_content(self, soup: BeautifulSoup) -> str:
        """Extract main text content from the page."""
        # Remove script and style elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer']):
            element.decompose()
        
        # Try to find main content areas
        content_selectors = [
            'main', 'article', '.content', '.main-content', 
            '.post-content', '.entry-content', '#content'
        ]
        
        for selector in content_selectors:
            content_area = soup.select_one(selector)
            if content_area:
                text = content_area.get_text(separator=' ', strip=True)
                if len(text) > 100:  # Minimum content length
                    return text[:2000]  # Limit text length
        
        # Fallback to body text
        body_text = soup.get_text(separator=' ', strip=True)
        return body_text[:2000] if body_text else "No content extracted"
    
    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, str]]:
        """Extract links from the page."""
        links = []
        for link in soup.find_all('a', href=True, limit=20):
            url = urljoin(base_url, link['href'])
            text = link.get_text(strip=True)
            if text and url.startswith(('http://', 'https://')):
                links.append({'url': url, 'text': text[:100]})
        return links
    
    def _extract_images(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract image URLs from the page."""
        images = []
        for img in soup.find_all('img', src=True, limit=10):
            img_url = urljoin(base_url, img['src'])
            if img_url.startswith(('http://', 'https://')):
                images.append(img_url)
        return images
    
    def _execute_command(self, command: str) -> Dict[str, Any]:
        """Execute a system command safely with timeout."""
        try:
            # Security check - only allow safe commands
            if not self._is_safe_command(command):
                return {
                    'success': False,
                    'error': f'Command not allowed for security reasons: {command}'
                }
            
            # Check cache
            cache_key = f"cmd_{command}"
            if self._is_cached(cache_key):
                cached_data = self.cache[cache_key]
                cached_data['from_cache'] = True
                return cached_data
            
            # Prepare command for execution
            if os.name == 'nt':  # Windows
                cmd_parts = ['cmd', '/c', command] if not command.startswith('powershell') else ['powershell', '-Command', command.replace('powershell ', '')]
            else:  # Unix-like
                cmd_parts = ['bash', '-c', command]
            
            # Execute command
            result = subprocess.run(
                cmd_parts,
                capture_output=True,
                text=True,
                timeout=self.command_timeout,
                cwd=os.getcwd()
            )
            
            data = {
                'success': result.returncode == 0,
                'command': command,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'return_code': result.returncode,
                'metadata': {
                    'execution_time': datetime.now().isoformat(),
                    'working_directory': os.getcwd()
                }
            }
            
            # Cache successful commands (non-sensitive ones)
            if data['success'] and self._is_cacheable_command(command):
                self._cache_data(cache_key, data, minutes=5)
            
            return data
            
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': f'Command timeout after {self.command_timeout} seconds'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Command execution error: {str(e)}'
            }
    
    def _is_safe_command(self, command: str) -> bool:
        """Check if a command is safe to execute."""
        command_lower = command.lower().strip()
        
        # Allowed command patterns
        safe_patterns = [
            r'^(dir|ls|pwd|cd|cat|type|more|head|tail|find|grep)\b',
            r'^(ping|nslookup|ipconfig|ifconfig|netstat)\b',
            r'^(systeminfo|uname|whoami|date|time|uptime)\b',
            r'^(tasklist|ps|top|htop)\b',
            r'^(python|node|java|git|npm|pip)\s+(--version|--help|-v|-h)\b',
            r'^powershell\s+(Get-Date|Get-Location|Get-Process|Get-Service)\b',
            r'^echo\s+',  # Allow echo commands
            r'^systeminfo\s*\|\s*findstr\s+'  # Allow systeminfo with findstr
        ]
        
        # Blocked patterns (dangerous commands)
        blocked_patterns = [
            r'\b(rm|del|delete|format|fdisk|mkfs)\b',
            r'\b(shutdown|reboot|halt|poweroff)\b',
            r'\b(sudo|su|runas)\b',
            r'\b(chmod|chown|attrib)\b',
            r'[;&|`$(){}]',  # Command injection characters
            r'\b(wget|curl)\s+.*\|\s*(sh|bash|cmd|powershell)\b'
        ]
        
        # Check if command is blocked
        for pattern in blocked_patterns:
            if re.search(pattern, command_lower):
                return False
        
        # Check if command matches safe patterns
        for pattern in safe_patterns:
            if re.search(pattern, command_lower):
                return True
        
        return False
    
    def _is_cacheable_command(self, command: str) -> bool:
        """Check if command output should be cached."""
        non_cacheable = ['date', 'time', 'ps', 'top', 'tasklist', 'netstat']
        command_lower = command.lower()
        return not any(cmd in command_lower for cmd in non_cacheable)
    
    def _search_web(self, query: str) -> Dict[str, Any]:
        """Perform a web search using multiple sources including Google."""
        try:
            # Try Google search first (using googlesearch-python if available)
            try:
                import googlesearch
                google_results = self._google_search(query)
                if google_results['success']:
                    return google_results
            except ImportError:
                # Fall back to other methods if googlesearch not available
                pass
            
            # Enhanced web scraping for sports and real-time data
            if any(term in query.lower() for term in ['cricket', 'score', 'live', 'match', 'india', 'pakistan']):
                return self._search_sports_websites(query)
            
            # Use DuckDuckGo as fallback
            search_url = "https://api.duckduckgo.com/"
            params = {
                'q': query,
                'format': 'json',
                'no_html': '1',
                'skip_disambig': '1'
            }
            
            response = self.session.get(search_url, params=params, timeout=self.web_timeout)
            response.raise_for_status()
            
            # Check if response is empty or invalid JSON
            if not response.text.strip():
                return {
                    'success': False,
                    'error': 'Empty response from search API'
                }
            
            try:
                data = response.json()
            except json.JSONDecodeError as e:
                # Fallback: return basic search info
                return {
                    'success': True,
                    'query': query,
                    'abstract': f"Search results for '{query}' - API returned non-JSON response",
                    'abstract_text': '',
                    'abstract_source': 'DuckDuckGo',
                    'abstract_url': '',
                    'definition': '',
                    'instant_answer': '',
                    'related_topics': [],
                    'infobox': {},
                    'metadata': {
                        'search_engine': 'DuckDuckGo',
                        'response_type': 'fallback',
                        'has_results': True,
                        'note': 'API response was not valid JSON'
                    }
                }
            
            return {
                'success': True,
                'query': query,
                'abstract': data.get('Abstract', ''),
                'abstract_text': data.get('AbstractText', ''),
                'abstract_source': data.get('AbstractSource', ''),
                'abstract_url': data.get('AbstractURL', ''),
                'definition': data.get('Definition', ''),
                'instant_answer': data.get('Answer', ''),
                'related_topics': [topic.get('Text', '') for topic in data.get('RelatedTopics', [])[:5]],
                'infobox': data.get('Infobox', {}),
                'metadata': {
                    'search_engine': 'DuckDuckGo',
                    'response_type': data.get('Type', ''),
                    'has_results': bool(data.get('Abstract') or data.get('Answer'))
                }
            }
            
        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'error': f'Search request error: {str(e)}'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Search error: {str(e)}'
            }
    
    def _google_search(self, query: str) -> Dict[str, Any]:
        """Perform Google search using googlesearch library."""
        try:
            from googlesearch import search
            results = []
            
            # Get top 5 results
            for url in search(query, num_results=5, lang='en'):
                results.append(url)
            
            if not results:
                return {
                    'success': False,
                    'error': 'No Google search results found'
                }
            
            # Scrape the first result for content
            first_result = self._retrieve_web_data(results[0])
            
            return {
                'success': True,
                'query': query,
                'search_results': results,
                'top_result_content': first_result if first_result.get('success') else None,
                'abstract': first_result.get('text_content', '')[:500] if first_result.get('success') else '',
                'metadata': {
                    'search_engine': 'Google',
                    'total_results': len(results),
                    'scraped_first': first_result.get('success', False)
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Google search error: {str(e)}'
            }
    
    def _search_sports_websites(self, query: str) -> Dict[str, Any]:
        """Search sports-specific websites for live scores and match info."""
        try:
            sports_urls = [
                'https://www.cricinfo.com',
                'https://www.cricbuzz.com', 
                'https://sports.ndtv.com/cricket',
                'https://www.espn.in/cricket/'
            ]
            
            # Create search queries
            search_terms = query.lower().replace(' ', '+')
            
            # Try Google search for sports
            google_query = f"site:cricinfo.com OR site:cricbuzz.com {query} live score"
            
            try:
                # Use requests to search Google directly
                google_search_url = f"https://www.google.com/search"
                params = {
                    'q': google_query,
                    'num': 5
                }
                
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                }
                
                response = self.session.get(google_search_url, params=params, headers=headers, timeout=self.web_timeout)
                
                if response.status_code == 200:
                    # Parse Google search results
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Look for sports score snippets
                    score_info = self._extract_sports_info(soup, query)
                    
                    if score_info:
                        return {
                            'success': True,
                            'query': query,
                            'sports_data': score_info,
                            'abstract': score_info.get('summary', ''),
                            'metadata': {
                                'source': 'Google Sports Search',
                                'search_method': 'sports_specific'
                            }
                        }
                
            except Exception as e:
                pass
            
            # Fallback: try direct website scraping
            for url in sports_urls[:2]:  # Try first 2 sports sites
                try:
                    result = self._retrieve_web_data(url)
                    if result.get('success'):
                        # Look for cricket/sports related content
                        content = result.get('text_content', '').lower()
                        if any(term in content for term in ['india', 'cricket', 'score', 'live']):
                            return {
                                'success': True,
                                'query': query,
                                'abstract': f"Sports information found on {url}",
                                'sports_content': result.get('text_content', '')[:500],
                                'metadata': {
                                    'source': url,
                                    'search_method': 'direct_scraping'
                                }
                            }
                except:
                    continue
            
            return {
                'success': False,
                'error': 'Could not retrieve sports information from available sources'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Sports search error: {str(e)}'
            }
    
    def _extract_sports_info(self, soup, query: str) -> Dict[str, Any]:
        """Extract sports information from Google search results."""
        try:
            sports_info = {}
            
            # Look for score boxes or sports snippets
            score_elements = soup.find_all(['div', 'span'], class_=lambda x: x and any(term in str(x).lower() for term in ['score', 'live', 'match', 'cricket']))
            
            if score_elements:
                for element in score_elements[:3]:
                    text = element.get_text(strip=True)
                    if text and len(text) > 10:
                        sports_info['summary'] = text[:200]
                        break
            
            # Look for structured data
            match_info = soup.find('div', class_='BNeawe')
            if match_info:
                sports_info['match_details'] = match_info.get_text(strip=True)
            
            return sports_info if sports_info else None
            
        except Exception as e:
            return None
    
    def _retrieve_api_data(self, query: str) -> Dict[str, Any]:
        """Retrieve data from various APIs based on query pattern."""
        query_lower = query.lower()
        
        # Sports data
        if any(sport in query_lower for sport in ['cricket', 'football', 'soccer', 'basketball', 'tennis', 'match', 'score', 'live']):
            return self._get_sports_data(query)
        
        # Weather data
        elif 'weather' in query_lower:
            return self._get_weather_data(query)
        
        # News data
        elif 'news' in query_lower:
            return self._get_news_data(query)
        
        # Default fallback
        return {
            'success': False,
            'error': f'No API handler for query: {query}'
        }
    
    def _get_weather_data(self, query: str) -> Dict[str, Any]:
        """Get weather data using a free weather API."""
        try:
            # Extract location from query
            location_match = re.search(r'weather\s+(?:in\s+|for\s+)?([a-zA-Z\s,]+)', query.lower())
            location = location_match.group(1).strip() if location_match else 'current location'
            
            # Use wttr.in API (free weather service)
            weather_url = f"https://wttr.in/{location}?format=j1"
            response = self.session.get(weather_url, timeout=self.web_timeout)
            
            if response.status_code == 200:
                weather_data = response.json()
                current = weather_data.get('current_condition', [{}])[0]
                
                return {
                    'success': True,
                    'location': location,
                    'current_weather': {
                        'temperature_c': current.get('temp_C'),
                        'temperature_f': current.get('temp_F'),
                        'condition': current.get('weatherDesc', [{}])[0].get('value'),
                        'humidity': current.get('humidity'),
                        'wind_speed': current.get('windspeedKmph'),
                        'wind_direction': current.get('winddir16Point')
                    },
                    'forecast': weather_data.get('weather', [])[:3],  # 3-day forecast
                    'metadata': {
                        'source': 'wttr.in',
                        'query': query
                    }
                }
            else:
                return {
                    'success': False,
                    'error': f'Weather API returned status {response.status_code}'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f'Weather data error: {str(e)}'
            }
    
    def _get_sports_data(self, query: str) -> Dict[str, Any]:
        """Get sports data primarily using Google search only."""
        try:
            # Use the improved sports search
            result = self._get_sports_from_google_only(query)
            
            # If no live data found, offer to open browser
            if not result.get('has_live_data', False):
                result['browser_fallback'] = True
                result['search_query'] = query
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Sports data error: {str(e)}',
                'browser_fallback': True,
                'search_query': query
            }
    
    def _get_sports_from_google_only(self, query: str) -> Dict[str, Any]:
        """Get sports data using multiple reliable approaches."""
        try:
            # First, try ESPN Cricinfo which is more reliable
            espn_result = self._check_espn_cricket(query)
            if espn_result.get('has_live_data'):
                return espn_result
            
            # Fallback to Cricbuzz
            cricbuzz_result = self._check_cricbuzz_simple(query)  
            if cricbuzz_result.get('has_live_data'):
                return cricbuzz_result
            
            # If nothing found, return honest response
            return {
                'success': True,
                'query': query,
                'abstract': 'Checked multiple sports sources but no live India vs Pakistan cricket match found at this moment.',
                'has_live_data': False,
                'metadata': {
                    'source': 'Multiple Sports Sources',
                    'verified_live': False,
                    'note': 'ESPN and Cricbuzz checked'
                }
            }
                
        except Exception as e:
            print(f"Error in sports search: {e}")
            return {
                'success': False,
                'query': query,
                'error': f'Sports search error: {str(e)}'
            }
    
    def _check_espn_cricket(self, query: str) -> Dict[str, Any]:
        """Check ESPN Cricinfo for live matches."""
        try:
            url = "https://www.espncricinfo.com/live-cricket-score"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = self.session.get(url, headers=headers, timeout=8)
            
            if response.status_code == 200:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(response.content, 'html.parser')
                page_text = soup.get_text().lower()
                
                # Check for specific teams in the query
                query_teams = []
                if 'india' in query.lower() or 'ind' in query.lower():
                    query_teams.append('india')
                if 'pakistan' in query.lower() or 'pak' in query.lower():
                    query_teams.append('pakistan')
                
                # Only return live data if we find the specific teams mentioned in query
                if query_teams:
                    teams_found = all(team in page_text for team in query_teams)
                    if teams_found and any(indicator in page_text for indicator in ['live', 'playing', 'batting', 'bowling']):
                        # Try to extract match info
                        title_tag = soup.find('title')
                        title = title_tag.get_text() if title_tag else "Live match found"
                        
                        return {
                            'success': True,
                            'query': query,
                            'abstract': f"Live cricket match found: {title[:200]}",
                            'has_live_data': True,
                            'metadata': {
                                'source': 'ESPN Cricinfo',
                                'verified_live': True,
                                'teams_found': query_teams
                            }
                        }
            
            return {'success': False, 'has_live_data': False}
            
        except Exception:
            return {'success': False, 'has_live_data': False}
    
    def _check_cricbuzz_simple(self, query: str) -> Dict[str, Any]:
        """Simple check of Cricbuzz for live matches."""
        try:
            url = "https://www.cricbuzz.com/"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = self.session.get(url, headers=headers, timeout=8)
            
            if response.status_code == 200:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(response.content, 'html.parser')
                page_text = soup.get_text().lower()
                
                # Check for specific teams in the query
                query_teams = []
                if 'india' in query.lower() or 'ind' in query.lower():
                    query_teams.append('india')
                if 'pakistan' in query.lower() or 'pak' in query.lower():
                    query_teams.append('pakistan')
                
                # Only return live data if we find the specific teams mentioned in query
                if query_teams:
                    teams_found = all(team in page_text for team in query_teams)
                    if teams_found and any(indicator in page_text for indicator in ['live', 'playing now', 'in progress']):
                        return {
                            'success': True,
                            'query': query,
                            'abstract': f'Live {" vs ".join(query_teams)} cricket match found on Cricbuzz',
                            'has_live_data': True,
                            'metadata': {
                                'source': 'Cricbuzz',
                                'verified_live': True,
                                'teams_found': query_teams
                            }
                        }
            
            return {'success': False, 'has_live_data': False}
            
        except Exception:
            return {'success': False, 'has_live_data': False}
    
    def _get_news_data(self, query: str) -> Dict[str, Any]:
        """Get news data using RSS feeds."""
        if not HAS_FEEDPARSER:
            return {
                'success': False,
                'error': 'Feedparser not available for news retrieval'
            }
        
        try:
            # Use public RSS feeds
            news_feeds = [
                'https://rss.cnn.com/rss/edition.rss',
                'https://feeds.bbci.co.uk/news/rss.xml',
                'https://rss.reuters.com/reuters/topNews'
            ]
            
            articles = []
            for feed_url in news_feeds[:1]:  # Limit to one feed for now
                try:
                    feed = feedparser.parse(feed_url)
                    for entry in feed.entries[:5]:  # Limit articles
                        articles.append({
                            'title': entry.get('title', ''),
                            'link': entry.get('link', ''),
                            'summary': entry.get('summary', ''),
                            'published': entry.get('published', ''),
                            'source': feed.feed.get('title', 'Unknown')
                        })
                except:
                    continue
            
            return {
                'success': True,
                'articles': articles,
                'query': query,
                'metadata': {
                    'source': 'RSS Feeds',
                    'article_count': len(articles)
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'News data error: {str(e)}'
            }
    
    def _is_cached(self, key: str) -> bool:
        """Check if data is cached and not expired."""
        if key not in self.cache:
            return False
        
        expiry_time = self.cache_expiry.get(key)
        if expiry_time and datetime.now() > expiry_time:
            del self.cache[key]
            del self.cache_expiry[key]
            return False
        
        return True
    
    def _cache_data(self, key: str, data: Dict[str, Any], minutes: int = 5):
        """Cache data with expiry time."""
        self.cache[key] = data.copy()
        self.cache_expiry[key] = datetime.now() + timedelta(minutes=minutes)
    
    def clear_cache(self):
        """Clear all cached data."""
        self.cache.clear()
        self.cache_expiry.clear()
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached data."""
        return {
            'total_cached_items': len(self.cache),
            'cache_keys': list(self.cache.keys()),
            'cache_sizes': {k: len(str(v)) for k, v in self.cache.items()},
            'memory_usage_bytes': sum(len(str(v)) for v in self.cache.values())
        }

    def _is_result_satisfactory(self, result: Dict[str, Any], original_query: str) -> bool:
        """Check if the retrieved result is satisfactory for the user query."""
        try:
            if not result.get('success', False):
                return False
            
            # Check if we have meaningful content
            abstract = result.get('abstract', '')
            text_content = result.get('text_content', '')
            stdout = result.get('stdout', '')
            instant_answer = result.get('instant_answer', '')
            
            # If there's substantial content, consider it satisfactory
            meaningful_content = [content for content in [abstract, text_content, stdout, instant_answer] if content and len(content.strip()) > 20]
            
            if len(meaningful_content) > 0:
                # Additional check: does the content seem relevant to the query?
                query_words = set(original_query.lower().split())
                for content in meaningful_content:
                    content_words = set(content.lower().split())
                    # If there's some overlap between query and content words, it's likely relevant
                    if len(query_words.intersection(content_words)) > 0:
                        return True
            
            # Check for error indicators
            if result.get('error') or 'error' in str(result).lower():
                return False
            
            # If we have search results but no content
            search_results = result.get('search_results', [])
            if len(search_results) == 0 and not meaningful_content:
                return False
                
            # Weather data check
            if 'weather' in original_query.lower():
                return bool(result.get('current_weather') or result.get('weather_data'))
            
            # News data check  
            if 'news' in original_query.lower():
                return bool(result.get('articles') or result.get('news_data'))
            
            # Sports data check
            if any(sport in original_query.lower() for sport in ['cricket', 'football', 'basketball', 'score', 'match']):
                return result.get('has_live_data', False) or bool(meaningful_content)
            
            # Default: if we got here and have some content, it's probably satisfactory
            return len(meaningful_content) > 0
            
        except Exception:
            return False

    def open_browser_search(self, query: str) -> bool:
        """Open default browser with search query."""
        try:
            import webbrowser
            import urllib.parse
            
            # Create search URL for Google
            search_url = f"https://www.google.com/search?q={urllib.parse.quote(query)}"
            
            # Open in default browser
            webbrowser.open(search_url)
            return True
            
        except Exception as e:
            print(f"Error opening browser: {e}")
            return False


# Global instance for easy import
data_retriever = DataRetriever()

# Convenience functions
def retrieve_data(query: str, method: str = "auto") -> Dict[str, Any]:
    """Convenience function to retrieve data."""
    return data_retriever.retrieve_data(query, method)

def scrape_web(url: str) -> Dict[str, Any]:
    """Convenience function for web scraping."""
    return data_retriever.retrieve_data(url, "web")

def execute_command(command: str) -> Dict[str, Any]:
    """Convenience function for command execution."""
    return data_retriever.retrieve_data(command, "command")

def search_web(query: str) -> Dict[str, Any]:
    """Convenience function for web search."""
    return data_retriever.retrieve_data(query, "search")

def get_weather(location: str = "") -> Dict[str, Any]:
    """Convenience function for weather data."""
    return data_retriever.retrieve_data(f"weather in {location}", "api")


if __name__ == "__main__":
    # Quick test
    retriever = DataRetriever()
    
    print("Testing Data Retrieval System...")
    
    # Test command execution
    result = retriever.retrieve_data("dir", "command")
    print(f"Command test success: {result['success']}")
    
    # Test web search
    result = retriever.retrieve_data("Python programming", "search")
    print(f"Search test success: {result['success']}")
    
    print("Data retrieval system ready!")