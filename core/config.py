"""
GaiaChat Configuration Module

Handles loading environment variables and providing configuration settings.
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class GaiaConfig:
    """Configuration settings for GaiaChat application."""
    
    # OpenAI Settings
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    model: str = field(
        default_factory=lambda: os.getenv("DEFAULT_LLM_MODEL", "gpt-4o-mini")
    )
    temperature: float = field(
        default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.7"))
    )
    max_tokens: int = field(
        default_factory=lambda: int(os.getenv("MAX_TOKENS", "4000"))
    )
    
    # Gaia Archive Settings
    gaia_tap_url: str = field(
        default_factory=lambda: os.getenv(
            "GAIA_TAP_URL", 
            "https://gea.esac.esa.int/tap-server/tap"
        )
    )
    gaia_table: str = field(
        default_factory=lambda: os.getenv("GAIA_TABLE", "gaiadr3.gaia_source")
    )
    
    # Query Limits
    max_query_results: int = field(
        default_factory=lambda: int(os.getenv("MAX_QUERY_RESULTS", "10000"))
    )
    default_query_results: int = field(
        default_factory=lambda: int(os.getenv("DEFAULT_QUERY_RESULTS", "1000"))
    )
    
    # Application Settings
    environment: str = field(
        default_factory=lambda: os.getenv("GAIA_ENV", "development")
    )
    log_level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO")
    )
    
    def validate(self) -> bool:
        """Validate that required configuration is present."""
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required. Set it in .env file.")
        return True


# Global configuration instance
config = GaiaConfig()
