#!/usr/bin/env python3
"""
Gemini Local Data Retrieval - AI-powered local system command generation and execution

This module uses Gemini AI to intelligently understand user queries about local data
and generate appropriate OS-specific commands to retrieve the information.

Features:
- Gemini AI understands user intent for local data needs
- Generates OS-specific commands (Windows, macOS, Linux)
- Safe command execution with validation
- Shared model architecture with the main assistant
- Intelligent result formatting and explanation

Author: Truvo Assistant
Version: 2.0 - Gemini AI Local Data Retrieval
"""

import os
import sys
import subprocess
import platform
from typing import Optional, Dict, Any, List
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

# OS Detection
try:
    from automation.os_detection import get_os_context, get_os_commands, os_detector
    OS_DETECTION_AVAILABLE = True
except ImportError:
    OS_DETECTION_AVAILABLE = False


class GeminiLocalDataRetriever:
    """Gemini AI-powered local data retrieval with intelligent command generation"""
    
    def __init__(self, gemini_client=None):
        """
        Initialize the Gemini Local Data Retriever
        
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
                print(f"Gemini Local Retriever using shared assistant model")
            else:
                # Create our own GeminiClient instance
                config = AssistantConfig(model_name="auto")
                self.gemini_client = GeminiClient(config=config)
                self.model = self.gemini_client.model
                self.model_name = "auto"
                print(f"Gemini Local Retriever initialized with model selection")
                
            # Get system information
            self.os_info = self._get_system_info()
            print(f"Detected OS: {self.os_info['os']} {self.os_info['version']}")
            
        except Exception as e:
            print(f"Error initializing Gemini Local Data Retriever: {e}")
            raise
    
    def _get_system_info(self) -> Dict[str, str]:
        """Get current system information"""
        try:
            return {
                'os': platform.system(),
                'version': platform.version(),
                'architecture': platform.architecture()[0],
                'machine': platform.machine(),
                'processor': platform.processor() or "Unknown",
                'node': platform.node()
            }
        except Exception:
            return {
                'os': 'Unknown',
                'version': 'Unknown',
                'architecture': 'Unknown',
                'machine': 'Unknown',
                'processor': 'Unknown',
                'node': 'Unknown'
            }
    
    def retrieve_local_data(self, query: str) -> str:
        """
        Use Gemini AI to understand query and generate appropriate local commands
        
        Args:
            query: User's local data query
            
        Returns:
            Formatted response with retrieved data
        """
        try:
            # Step 1: Get Gemini to generate appropriate commands
            commands = self._generate_commands(query)
            
            if not commands:
                return "I couldn't determine appropriate commands for your request."
            
            # Step 2: Execute commands safely
            results = []
            for cmd_info in commands:
                result = self._execute_command_safely(cmd_info)
                results.append(result)
            
            # Step 3: Format and present results
            return self._format_results(query, results)
            
        except Exception as e:
            print(f"Gemini local data retrieval error: {e}")
            return f"I encountered an error while retrieving local data: {e}"
    
    def _generate_commands(self, query: str) -> List[Dict[str, Any]]:
        """Generate appropriate OS-specific commands using Gemini AI"""
        try:
            # Create a detailed prompt for command generation
            prompt = f"""You are an intelligent local system assistant. The user wants local data/information from their computer.

User Query: "{query}"
Operating System: {self.os_info['os']} {self.os_info['version']}
Architecture: {self.os_info['architecture']}

Your task is to generate appropriate, SAFE READ-ONLY commands to retrieve the requested information. 

IMPORTANT SAFETY RULES:
- Only use READ-ONLY commands (no rm, del, format, etc.)
- No commands that modify, delete, or damage files/system
- No network commands unless specifically needed for data retrieval
- No commands requiring admin/sudo unless absolutely necessary
- Focus on data retrieval and information gathering only

Respond with a JSON array of command objects. Each object should have:
{{
    "command": "actual command to run",
    "description": "what this command does",
    "os_specific": "windows/macos/linux/all",
    "safe": true/false,
    "explanation": "why this command is appropriate"
}}

Examples of SAFE commands:
- Windows: dir, type, findstr, wmic, tasklist, systeminfo, powershell Get-*
- macOS/Linux: ls, cat, grep, find, ps, df, du, whoami, top

Examples for different queries:
- "find my Python files" → find/dir commands with .py filter
- "check disk space" → df/dir commands
- "show running processes" → ps/tasklist commands
- "find large files" → find/forfiles with size filters
- "system information" → systeminfo/uname commands

Generate 1-3 commands maximum. Focus on the most relevant and safe approach for data retrieval.

Commands:"""

            # Generate response from Gemini
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Extract JSON from response
            try:
                # Find JSON array in response
                json_start = response_text.find('[')
                json_end = response_text.rfind(']') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    commands = json.loads(json_str)
                    
                    # Filter for safe commands only
                    safe_commands = [cmd for cmd in commands if cmd.get('safe', False)]
                    return safe_commands
                else:
                    print(f"No valid JSON found in response: {response_text}")
                    return []
                    
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                print(f"Response: {response_text}")
                return []
            
        except Exception as e:
            print(f"Error generating commands: {e}")
            return []
    
    def _execute_command_safely(self, cmd_info: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a command safely with validation for data retrieval only"""
        try:
            command = cmd_info.get('command', '')
            description = cmd_info.get('description', 'Unknown command')
            
            # Safety check - only allow read-only commands for data retrieval
            dangerous_keywords = ['rm', 'del', 'format', 'rmdir', 'rd', 'kill', 'shutdown', 'reboot']
            if any(keyword in command.lower() for keyword in dangerous_keywords):
                return {
                    'success': False,
                    'error': f"Command rejected for safety (data retrieval only): {command}",
                    'description': description
                }
            
            print(f"Executing: {command}")
            
            # Execute command with timeout
            if self.os_info['os'] == 'Windows':
                result = subprocess.run(
                    command, 
                    shell=True, 
                    capture_output=True, 
                    text=True, 
                    timeout=30,
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
            else:
                result = subprocess.run(
                    command, 
                    shell=True, 
                    capture_output=True, 
                    text=True, 
                    timeout=30
                )
            
            return {
                'success': result.returncode == 0,
                'output': result.stdout if result.returncode == 0 else result.stderr,
                'command': command,
                'description': description,
                'return_code': result.returncode
            }
            
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': 'Command timed out after 30 seconds',
                'command': command,
                'description': description
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'command': command,
                'description': description
            }
    
    def _format_results(self, query: str, results: List[Dict[str, Any]]) -> str:
        """Format command results into a user-friendly response"""
        try:
            # Create prompt for result formatting
            results_summary = []
            for result in results:
                if result['success']:
                    results_summary.append({
                        'command': result['command'],
                        'description': result['description'],
                        'output': result['output'][:1000] + "..." if len(result['output']) > 1000 else result['output']
                    })
                else:
                    results_summary.append({
                        'command': result['command'],
                        'description': result['description'],
                        'error': result.get('error', 'Unknown error')
                    })
            
            format_prompt = f"""Format the following command execution results in a user-friendly way for the query: "{query}"

Results:
{json.dumps(results_summary, indent=2)}

Please provide a clear, well-formatted response that:
1. Directly answers the user's question
2. Presents the information in an organized way
3. Explains what was found
4. Uses appropriate formatting (bullet points, headers, etc.)
5. Highlights key information
6. If there were errors, explain them clearly

Response:"""

            response = self.model.generate_content(format_prompt)
            return response.text.strip()
            
        except Exception as e:
            # Fallback formatting
            formatted = f"Results for: {query}\n" + "="*50 + "\n\n"
            for i, result in enumerate(results, 1):
                if result['success']:
                    formatted += f"Command {i}: {result['description']}\n"
                    formatted += f"Output: {result['output'][:500]}...\n\n"
                else:
                    formatted += f"Command {i} failed: {result['description']}\n"
                    formatted += f"Error: {result.get('error', 'Unknown error')}\n\n"
            return formatted


def retrieve_data(query: str, gemini_client=None) -> str:
    """
    Main function for local data retrieval using Gemini AI
    
    Args:
        query: Local data query
        gemini_client: Optional shared GeminiClient instance
        
    Returns:
        Retrieved and formatted local data
    """
    try:
        retriever = GeminiLocalDataRetriever(gemini_client=gemini_client)
        return retriever.retrieve_local_data(query)
    except Exception as e:
        return f"Local data retrieval failed: {e}"


# Test function removed - module is integrated into main system