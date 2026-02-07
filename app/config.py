"""
Configuration for Topic Classifier Service
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # Service Configuration
    SERVICE_NAME: str = "topic-classifier-service"
    SERVICE_PORT: int = 8009
    HOST: str = "0.0.0.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # Database Configuration
    DATABASE_URL: str = "postgresql://modai:modai@localhost:5432/modai"

    # Google Gemini API
    GOOGLE_API_KEY: str
    GEMINI_MODEL: str = "gemini-2.0-flash-exp"  # Smallest, fastest Gemini model
    GEMINI_TEMPERATURE: float = 0.1  # Low temperature for consistent classification
    GEMINI_MAX_TOKENS: int = 500

    # Redis Configuration
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 3  # Use DB 3 for topic classifier
    REDIS_CACHE_TTL: int = 3600  # 1 hour cache

    # Classification Settings
    CONFIDENCE_THRESHOLD: float = 0.6  # Minimum confidence for topic detection
    MAX_TOPICS_PER_MESSAGE: int = 5
    ENABLE_CACHING: bool = True

    # Classification Method
    CLASSIFICATION_METHOD: str = "onnx"  # "onnx" (ONNX Runtime), "bart" (PyTorch), or "claude" (Anthropic API)
    BART_MODEL: str = "valhalla/distilbart-mnli-12-1"  # Distilled BART model (~400MB)
    ONNX_MODEL: str = "./models/deberta-onnx"  # Local ONNX model path

    # Anthropic Claude API
    ANTHROPIC_API_KEY: str = ""
    CLAUDE_MODEL: str = "claude-3-5-sonnet-20241022"  # Claude 3.5 Sonnet (fast, accurate)
    CLAUDE_TEMPERATURE: float = 0.1  # Low temperature for consistent classification
    CLAUDE_MAX_TOKENS: int = 500

    # Model Server (for shared ONNX inference)
    MODEL_SERVER_URL: str = "http://topic-model-server:8010"

    # Taxonomy Path
    TAXONOMY_MODULE_PATH: str = "../../models/TOPIC_TAXONOMY.py"

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # Allow extra env vars for monolith mode


# Global settings instance
settings = Settings()
