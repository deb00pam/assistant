"""
Core Module - Business logic and orchestration for Truvo

This module contains the core business logic, orchestration components,
and configuration.
"""

from .config import AssistantConfig, setup_environment

# Lazy import of DesktopAssistant to avoid circular dependencies during startup
def __getattr__(name):
    if name == 'DesktopAssistant':
        from .desktop_assistant import DesktopAssistant
        return DesktopAssistant
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = ['DesktopAssistant', 'AssistantConfig', 'setup_environment']
