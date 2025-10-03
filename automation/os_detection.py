#!/usr/bin/env python3
"""
OS Detection Module for Truvo Desktop Assistant

Detects the operating system and provides OS-specific information
to help Gemini provide appropriate responses and commands.
"""

import os
import platform
import sys
import subprocess
import winreg
from pathlib import Path
from typing import Dict, Any, Optional, List

class OSDetector:
    """Detects and provides information about the current operating system."""
    
    def __init__(self):
        self.os_info = self._detect_os()
        self.system_blueprint = self._create_system_blueprint()
    
    def _detect_os(self) -> Dict[str, Any]:
        """Detect comprehensive OS information."""
        try:
            # Basic OS detection
            system = platform.system()
            release = platform.release()
            version = platform.version()
            machine = platform.machine()
            processor = platform.processor()
            
            # Python environment info
            python_version = platform.python_version()
            python_implementation = platform.python_implementation()
            
            # More specific OS detection
            is_windows = system == "Windows"
            is_macos = system == "Darwin"
            is_linux = system == "Linux"
            
            # Windows-specific detection
            windows_edition = None
            if is_windows:
                try:
                    import winreg
                    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 
                                      r"SOFTWARE\Microsoft\Windows NT\CurrentVersion") as key:
                        windows_edition = winreg.QueryValueEx(key, "ProductName")[0]
                except:
                    windows_edition = f"Windows {release}"
            
            # Linux distribution detection
            linux_distro = None
            if is_linux:
                try:
                    # Try using platform.freedesktop_os_release() for newer Python versions
                    if hasattr(platform, 'freedesktop_os_release'):
                        distro_info = platform.freedesktop_os_release()
                        linux_distro = f"{distro_info.get('NAME', 'Linux')} {distro_info.get('VERSION', '')}"
                    else:
                        # Fallback for older Python versions
                        linux_distro = platform.linux_distribution()[0] if hasattr(platform, 'linux_distribution') else "Linux"
                except:
                    linux_distro = "Linux Distribution"
            
            # macOS version detection
            macos_version = None
            if is_macos:
                try:
                    mac_ver = platform.mac_ver()
                    macos_version = f"macOS {mac_ver[0]}"
                except:
                    macos_version = "macOS"
            
            # Determine shell/terminal
            shell = os.environ.get('SHELL', 'Unknown')
            if is_windows:
                # Check for common Windows shells
                if 'powershell' in os.environ.get('PSModulePath', '').lower():
                    shell = 'PowerShell'
                elif os.environ.get('COMSPEC', '').endswith('cmd.exe'):
                    shell = 'Command Prompt (cmd)'
                else:
                    shell = 'Windows Shell'
            
            return {
                'system': system,
                'release': release,
                'version': version,
                'machine': machine,
                'processor': processor,
                'is_windows': is_windows,
                'is_macos': is_macos,
                'is_linux': is_linux,
                'windows_edition': windows_edition,
                'linux_distro': linux_distro,
                'macos_version': macos_version,
                'shell': shell,
                'python_version': python_version,
                'python_implementation': python_implementation,
                'architecture': platform.architecture()[0],
                'node': platform.node()  # Computer name
            }
        
        except Exception as e:
            # Fallback basic detection
            return {
                'system': platform.system(),
                'release': platform.release(),
                'is_windows': platform.system() == "Windows",
                'is_macos': platform.system() == "Darwin", 
                'is_linux': platform.system() == "Linux",
                'error': str(e)
            }
    
    def _create_system_blueprint(self) -> Dict[str, Any]:
        """Create a comprehensive blueprint of the system's capabilities."""
        try:
            blueprint = {
                'applications': self._detect_applications(),
                'settings_locations': self._get_settings_locations(),
                'file_locations': self._get_important_file_locations(),
                'system_capabilities': self._detect_system_capabilities(),
                'installed_software': self._detect_installed_software(),
                'browser_info': self._detect_browsers(),
                'development_tools': self._detect_dev_tools()
            }
            return blueprint
        except Exception as e:
            # Fallback for any system blueprint creation errors
            return {
                'error': f"Failed to create system blueprint: {str(e)}",
                'basic_info_available': True
            }
    
    def _detect_applications(self) -> Dict[str, List[str]]:
        """Detect available applications and their common names."""
        apps = {
            'built_in': [],
            'microsoft_office': [],
            'browsers': [],
            'media': [],
            'development': [],
            'utilities': [],
            'games': []
        }
        
        if self.os_info['is_windows']:
            # Windows built-in applications
            apps['built_in'] = [
                'notepad', 'calculator', 'paint', 'wordpad', 'snipping tool',
                'task manager', 'file explorer', 'settings', 'control panel',
                'command prompt', 'powershell', 'registry editor', 'device manager'
            ]
            
            # Check for common applications in registry and file system
            common_apps = {
                'microsoft_office': [
                    ('Microsoft Office\\root\\Office16\\WINWORD.EXE', 'Microsoft Word'),
                    ('Microsoft Office\\root\\Office16\\EXCEL.EXE', 'Microsoft Excel'),
                    ('Microsoft Office\\root\\Office16\\POWERPNT.EXE', 'Microsoft PowerPoint'),
                    ('Microsoft Office\\root\\Office16\\OUTLOOK.EXE', 'Microsoft Outlook')
                ],
                'browsers': [
                    ('Google\\Chrome\\Application\\chrome.exe', 'Google Chrome'),
                    ('Mozilla Firefox\\firefox.exe', 'Mozilla Firefox'),
                    ('Microsoft\\Edge\\Application\\msedge.exe', 'Microsoft Edge'),
                    ('BraveSoftware\\Brave-Browser\\Application\\brave.exe', 'Brave Browser')
                ],
                'media': [
                    ('Windows Media Player\\wmplayer.exe', 'Windows Media Player'),
                    ('VideoLAN\\VLC\\vlc.exe', 'VLC Media Player'),
                    ('Spotify\\Spotify.exe', 'Spotify'),
                    ('foobar2000\\foobar2000.exe', 'foobar2000')
                ],
                'development': [
                    ('Microsoft VS Code\\Code.exe', 'Visual Studio Code'),
                    ('JetBrains\\PyCharm\\bin\\pycharm64.exe', 'PyCharm'),
                    ('Git\\bin\\git.exe', 'Git'),
                    ('Python\\python.exe', 'Python')
                ]
            }
            
            # Check Program Files and AppData for installed apps
            search_paths = [
                Path(os.environ.get('PROGRAMFILES', 'C:\\Program Files')),
                Path(os.environ.get('PROGRAMFILES(X86)', 'C:\\Program Files (x86)')),
                Path(os.environ.get('LOCALAPPDATA', '')) / 'Programs'
            ]
            
            for category, app_list in common_apps.items():
                for app_path, app_name in app_list:
                    found = False
                    for search_path in search_paths:
                        if search_path.exists():
                            full_path = search_path / app_path
                            if full_path.exists():
                                apps[category].append(app_name)
                                found = True
                                break
                    
                    # Also check Windows Store apps via registry
                    if not found and category == 'browsers':
                        try:
                            # Check for Edge WebView or Windows Store versions
                            if 'edge' in app_name.lower():
                                apps[category].append('Microsoft Edge')
                        except:
                            pass
        
        return apps
    
    def _get_settings_locations(self) -> Dict[str, str]:
        """Get locations of various system settings."""
        settings = {}
        
        if self.os_info['is_windows']:
            settings = {
                'windows_settings': 'ms-settings:',
                'control_panel': 'control',
                'device_manager': 'devmgmt.msc',
                'disk_management': 'diskmgmt.msc',
                'services': 'services.msc',
                'registry_editor': 'regedit',
                'system_configuration': 'msconfig',
                'task_manager': 'taskmgr',
                'network_settings': 'ms-settings:network',
                'display_settings': 'ms-settings:display',
                'sound_settings': 'ms-settings:sound',
                'privacy_settings': 'ms-settings:privacy',
                'update_settings': 'ms-settings:windowsupdate',
                'apps_settings': 'ms-settings:appsfeatures',
                'storage_settings': 'ms-settings:storagesense'
            }
        elif self.os_info['is_macos']:
            settings = {
                'system_preferences': 'System Preferences',
                'network_preferences': 'Network Preferences',
                'security_privacy': 'Security & Privacy',
                'displays': 'Displays',
                'sound': 'Sound',
                'users_groups': 'Users & Groups'
            }
        elif self.os_info['is_linux']:
            settings = {
                'system_settings': 'gnome-control-center',
                'network_settings': 'nm-connection-editor',
                'display_settings': 'gnome-display-properties',
                'sound_settings': 'gnome-sound-properties'
            }
        
        return settings
    
    def _get_important_file_locations(self) -> Dict[str, str]:
        """Get important file and folder locations."""
        locations = {}
        
        if self.os_info['is_windows']:
            locations = {
                'desktop': os.path.join(os.environ.get('USERPROFILE', ''), 'Desktop'),
                'documents': os.path.join(os.environ.get('USERPROFILE', ''), 'Documents'),
                'downloads': os.path.join(os.environ.get('USERPROFILE', ''), 'Downloads'),
                'pictures': os.path.join(os.environ.get('USERPROFILE', ''), 'Pictures'),
                'music': os.path.join(os.environ.get('USERPROFILE', ''), 'Music'),
                'videos': os.path.join(os.environ.get('USERPROFILE', ''), 'Videos'),
                'startup_folder': os.path.join(os.environ.get('APPDATA', ''), 'Microsoft\\Windows\\Start Menu\\Programs\\Startup'),
                'temp_folder': os.environ.get('TEMP', ''),
                'program_files': os.environ.get('PROGRAMFILES', ''),
                'system32': os.path.join(os.environ.get('WINDIR', ''), 'System32')
            }
        elif self.os_info['is_macos']:
            home = os.path.expanduser('~')
            locations = {
                'desktop': os.path.join(home, 'Desktop'),
                'documents': os.path.join(home, 'Documents'),
                'downloads': os.path.join(home, 'Downloads'),
                'pictures': os.path.join(home, 'Pictures'),
                'music': os.path.join(home, 'Music'),
                'movies': os.path.join(home, 'Movies'),
                'applications': '/Applications',
                'library': os.path.join(home, 'Library')
            }
        elif self.os_info['is_linux']:
            home = os.path.expanduser('~')
            locations = {
                'desktop': os.path.join(home, 'Desktop'),
                'documents': os.path.join(home, 'Documents'),
                'downloads': os.path.join(home, 'Downloads'),
                'pictures': os.path.join(home, 'Pictures'),
                'music': os.path.join(home, 'Music'),
                'videos': os.path.join(home, 'Videos'),
                'applications': '/usr/share/applications',
                'bin': '/usr/bin'
            }
        
        return locations
    
    def _detect_system_capabilities(self) -> Dict[str, Any]:
        """Detect system capabilities and features."""
        capabilities = {
            'has_touchscreen': False,
            'has_microphone': False,
            'has_camera': False,
            'has_bluetooth': False,
            'has_wifi': False,
            'display_count': 1,
            'cpu_cores': os.cpu_count(),
            'supports_virtualization': False
        }
        
        if self.os_info['is_windows']:
            try:
                # Check for touchscreen
                import ctypes
                SM_DIGITIZER = 94
                capabilities['has_touchscreen'] = ctypes.windll.user32.GetSystemMetrics(SM_DIGITIZER) > 0
                
                # Check for audio devices
                result = subprocess.run(['powershell', '-Command', 
                    'Get-WmiObject -Class Win32_SoundDevice | Select-Object Name'], 
                    capture_output=True, text=True, timeout=5)
                capabilities['has_microphone'] = 'microphone' in result.stdout.lower() or len(result.stdout.strip()) > 50
                
                # Check for cameras
                result = subprocess.run(['powershell', '-Command',
                    'Get-WmiObject -Class Win32_PnPEntity | Where-Object {$_.Name -like "*camera*"} | Select-Object Name'],
                    capture_output=True, text=True, timeout=5)
                capabilities['has_camera'] = len(result.stdout.strip()) > 50
                
                # Check display count
                result = subprocess.run(['powershell', '-Command',
                    '(Get-WmiObject -Class Win32_VideoController | Where-Object {$_.Status -eq "OK"}).Count'],
                    capture_output=True, text=True, timeout=5)
                try:
                    capabilities['display_count'] = int(result.stdout.strip())
                except:
                    capabilities['display_count'] = 1
                
            except Exception as e:
                # Silent fail - capabilities detection is optional
                pass
        
        return capabilities
    
    def _detect_installed_software(self) -> Dict[str, List[str]]:
        """Detect major installed software categories."""
        software = {
            'productivity': [],
            'creative': [],
            'gaming': [],
            'communication': [],
            'development': [],
            'security': []
        }
        
        if self.os_info['is_windows']:
            # Check registry for installed programs
            try:
                registry_paths = [
                    r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall",
                    r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall"
                ]
                
                software_keywords = {
                    'productivity': ['office', 'word', 'excel', 'powerpoint', 'outlook', 'onenote', 'acrobat', 'reader'],
                    'creative': ['photoshop', 'illustrator', 'premiere', 'after effects', 'gimp', 'blender', 'audacity'],
                    'gaming': ['steam', 'origin', 'epic games', 'battle.net', 'xbox', 'nvidia geforce'],
                    'communication': ['zoom', 'teams', 'skype', 'discord', 'slack', 'whatsapp'],
                    'development': ['visual studio', 'pycharm', 'intellij', 'eclipse', 'git', 'docker', 'nodejs'],
                    'security': ['windows defender', 'norton', 'mcafee', 'avast', 'malwarebytes']
                }
                
                for registry_path in registry_paths:
                    try:
                        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, registry_path) as key:
                            i = 0
                            while True:
                                try:
                                    subkey_name = winreg.EnumKey(key, i)
                                    with winreg.OpenKey(key, subkey_name) as subkey:
                                        try:
                                            display_name = winreg.QueryValueEx(subkey, "DisplayName")[0].lower()
                                            for category, keywords in software_keywords.items():
                                                for keyword in keywords:
                                                    if keyword in display_name and display_name not in software[category]:
                                                        software[category].append(display_name.title())
                                        except FileNotFoundError:
                                            pass
                                    i += 1
                                except OSError:
                                    break
                    except Exception:
                        continue
                        
            except Exception as e:
                # Silent fail
                pass
        
        return software
    
    def _detect_browsers(self) -> Dict[str, Any]:
        """Detect installed browsers and their details."""
        browsers = {
            'installed': [],
            'default': None,
            'profiles': {}
        }
        
        if self.os_info['is_windows']:
            # Check for common browser executables
            browser_paths = {
                'Chrome': [
                    r'Google\Chrome\Application\chrome.exe',
                    r'Google\Chrome Beta\Application\chrome.exe'
                ],
                'Firefox': [r'Mozilla Firefox\firefox.exe'],
                'Edge': [r'Microsoft\Edge\Application\msedge.exe'],
                'Brave': [r'BraveSoftware\Brave-Browser\Application\brave.exe'],
                'Opera': [r'Opera\launcher.exe']
            }
            
            search_locations = [
                os.environ.get('PROGRAMFILES', ''),
                os.environ.get('PROGRAMFILES(X86)', ''),
                os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Programs')
            ]
            
            for browser_name, paths in browser_paths.items():
                for browser_path in paths:
                    for location in search_locations:
                        full_path = os.path.join(location, browser_path)
                        if os.path.exists(full_path):
                            browsers['installed'].append(browser_name)
                            break
            
            # Try to detect default browser
            try:
                result = subprocess.run(['powershell', '-Command',
                    'Get-ItemProperty "HKCU:\\Software\\Microsoft\\Windows\\Shell\\Associations\\UrlAssociations\\http\\UserChoice" | Select-Object ProgId'],
                    capture_output=True, text=True, timeout=5)
                
                default_browser_map = {
                    'ChromeHTML': 'Chrome',
                    'FirefoxURL': 'Firefox',
                    'MSEdgeHTM': 'Edge',
                    'BraveHTML': 'Brave'
                }
                
                for prog_id, browser in default_browser_map.items():
                    if prog_id in result.stdout:
                        browsers['default'] = browser
                        break
                        
            except Exception:
                pass
        
        return browsers
    
    def _detect_dev_tools(self) -> Dict[str, List[str]]:
        """Detect development tools and environments."""
        dev_tools = {
            'editors': [],
            'languages': [],
            'databases': [],
            'version_control': [],
            'containers': []
        }
        
        # Check for common development tools
        if self.os_info['is_windows']:
            dev_paths = {
                'editors': [
                    ('Microsoft VS Code\\Code.exe', 'Visual Studio Code'),
                    ('Sublime Text 3\\sublime_text.exe', 'Sublime Text'),
                    ('Notepad++\\notepad++.exe', 'Notepad++')
                ],
                'languages': [
                    ('Python\\python.exe', 'Python'),
                    ('nodejs\\node.exe', 'Node.js'),
                    ('Java\\jdk\\bin\\java.exe', 'Java'),
                    ('Go\\bin\\go.exe', 'Go')
                ],
                'version_control': [
                    ('Git\\bin\\git.exe', 'Git'),
                    ('TortoiseGit\\bin\\TortoiseGitProc.exe', 'TortoiseGit')
                ]
            }
            
            search_paths = [
                os.environ.get('PROGRAMFILES', ''),
                os.environ.get('PROGRAMFILES(X86)', ''),
                os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Programs')
            ]
            
            for category, tools in dev_paths.items():
                for tool_path, tool_name in tools:
                    for search_path in search_paths:
                        full_path = os.path.join(search_path, tool_path)
                        if os.path.exists(full_path):
                            dev_tools[category].append(tool_name)
                            break
        
        return dev_tools
    
    def get_os_context_for_ai(self) -> str:
        """Generate OS context string to inform AI about the current environment."""
        info = self.os_info
        blueprint = self.system_blueprint
        
        # Build context string
        context_parts = []
        
        if info['is_windows']:
            os_name = info.get('windows_edition', f"Windows {info['release']}")
            context_parts.append(f"Operating System: {os_name}")
            context_parts.append("Shell: PowerShell (default Windows terminal)")
            context_parts.append("Commands: Use Windows-specific commands (dir, copy, move, etc.)")
            context_parts.append("File paths: Use backslashes (\\) and Windows path format")
            context_parts.append("Applications: Reference Windows applications and Control Panel")
            
            # Add system blueprint information
            if blueprint and 'error' not in blueprint:
                # Add available applications
                apps = blueprint.get('applications', {})
                if apps.get('built_in'):
                    context_parts.append(f"Built-in apps available: {', '.join(apps['built_in'][:5])}...")
                
                if apps.get('browsers'):
                    context_parts.append(f"Browsers installed: {', '.join(apps['browsers'])}")
                
                # Add settings locations
                settings = blueprint.get('settings_locations', {})
                context_parts.append("Settings can be accessed via: Windows Settings (ms-settings:), Control Panel (control), Device Manager (devmgmt.msc)")
                
                # Add system capabilities
                capabilities = blueprint.get('system_capabilities', {})
                cap_list = []
                if capabilities.get('has_touchscreen'):
                    cap_list.append("touchscreen")
                if capabilities.get('has_microphone'):
                    cap_list.append("microphone")
                if capabilities.get('has_camera'):
                    cap_list.append("camera")
                if capabilities.get('cpu_cores', 0) > 0:
                    cap_list.append(f"{capabilities['cpu_cores']} CPU cores")
                
                if cap_list:
                    context_parts.append(f"System capabilities: {', '.join(cap_list)}")
            
        elif info['is_macos']:
            os_name = info.get('macos_version', 'macOS')
            context_parts.append(f"Operating System: {os_name}")
            context_parts.append("Shell: Unix/bash terminal")
            context_parts.append("Commands: Use Unix/macOS commands (ls, cp, mv, etc.)")
            context_parts.append("File paths: Use forward slashes (/) and Unix path format")
            context_parts.append("Applications: Reference macOS applications and System Preferences")
            
        elif info['is_linux']:
            os_name = info.get('linux_distro', 'Linux')
            context_parts.append(f"Operating System: {os_name}")
            context_parts.append("Shell: Unix/bash terminal")
            context_parts.append("Commands: Use Linux commands (ls, cp, mv, apt/yum, etc.)")
            context_parts.append("File paths: Use forward slashes (/) and Unix path format")
            context_parts.append("Applications: Reference Linux applications and package managers")
            
        else:
            context_parts.append(f"Operating System: {info['system']} {info['release']}")
            context_parts.append("Commands: Use appropriate commands for this OS")
        
        # Add common info
        context_parts.append(f"Architecture: {info.get('architecture', 'Unknown')}")
        context_parts.append(f"Python: {info['python_version']}")
        
        # Add system blueprint summary if available
        if hasattr(self, 'system_blueprint') and self.system_blueprint and 'error' not in self.system_blueprint:
            blueprint = self.system_blueprint
            
            # Add file locations
            file_locs = blueprint.get('file_locations', {})
            if file_locs:
                key_folders = ['desktop', 'documents', 'downloads']
                available_folders = [folder for folder in key_folders if folder in file_locs]
                if available_folders:
                    context_parts.append(f"Key folders: {', '.join(available_folders)}")
            
            # Add development tools summary
            dev_tools = blueprint.get('development_tools', {})
            if dev_tools:
                tools_summary = []
                if dev_tools.get('editors'):
                    tools_summary.append(f"Editors: {', '.join(dev_tools['editors'][:2])}")
                if dev_tools.get('languages'):
                    tools_summary.append(f"Languages: {', '.join(dev_tools['languages'][:3])}")
                if tools_summary:
                    context_parts.append(f"Dev tools: {'; '.join(tools_summary)}")
        
        return " | ".join(context_parts)
    
    def get_os_specific_commands(self) -> Dict[str, str]:
        """Get OS-specific command equivalents."""
        if self.os_info['is_windows']:
            return {
                'list_files': 'dir',
                'copy_file': 'copy',
                'move_file': 'move',
                'delete_file': 'del',
                'make_directory': 'mkdir',
                'change_directory': 'cd',
                'current_directory': 'cd',
                'clear_screen': 'cls',
                'find_text': 'findstr',
                'process_list': 'tasklist',
                'kill_process': 'taskkill',
                'network_info': 'ipconfig',
                'path_separator': '\\',
                'line_ending': '\\r\\n'
            }
        else:  # Unix-like (Linux, macOS)
            return {
                'list_files': 'ls',
                'copy_file': 'cp',
                'move_file': 'mv',
                'delete_file': 'rm',
                'make_directory': 'mkdir',
                'change_directory': 'cd',
                'current_directory': 'pwd',
                'clear_screen': 'clear',
                'find_text': 'grep',
                'process_list': 'ps',
                'kill_process': 'kill',
                'network_info': 'ifconfig',
                'path_separator': '/',
                'line_ending': '\\n'
            }
    
    def get_detailed_info(self) -> Dict[str, Any]:
        """Get all detected OS information."""
        return self.os_info.copy()
    
    def __str__(self) -> str:
        """String representation of OS info."""
        info = self.os_info
        if info['is_windows']:
            return info.get('windows_edition', f"Windows {info['release']}")
        elif info['is_macos']:
            return info.get('macos_version', 'macOS')
        elif info['is_linux']:
            return info.get('linux_distro', 'Linux')
        else:
            return f"{info['system']} {info['release']}"


# Global OS detector instance
os_detector = OSDetector()

def get_os_context() -> str:
    """Quick function to get OS context for AI."""
    return os_detector.get_os_context_for_ai()

def get_os_commands() -> Dict[str, str]:
    """Quick function to get OS-specific commands."""
    return os_detector.get_os_specific_commands()


if __name__ == "__main__":
    # Test the OS detection
    detector = OSDetector()
    print("=== OS Detection Results ===")
    print(f"OS: {detector}")
    print(f"AI Context: {detector.get_os_context_for_ai()}")
    print("\n=== OS-Specific Commands ===")
    commands = detector.get_os_specific_commands()
    for action, command in commands.items():
        print(f"{action}: {command}")
    print("\n=== Detailed Info ===")
    import json
    print(json.dumps(detector.get_detailed_info(), indent=2))