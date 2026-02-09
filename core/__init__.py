"""
Core module for GaiaChat.
"""

from .config import config, GaiaConfig
from .gaia_service import gaia_service, GaiaService, QueryResult
from .agent import agent, GaiaChatAgent, AgentResponse

__all__ = [
    'config', 'GaiaConfig',
    'gaia_service', 'GaiaService', 'QueryResult',
    'agent', 'GaiaChatAgent', 'AgentResponse'
]
