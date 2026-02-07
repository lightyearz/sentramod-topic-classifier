"""
Domain models for Topic Classification
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
from enum import IntEnum
from datetime import datetime


class Tier(IntEnum):
    """Safety tier enumeration"""

    GREEN = 1  # Academic, safe topics
    YELLOW = 2  # Sensitive topics requiring approval
    ORANGE = 3  # Mental health concerns requiring monitoring
    RED = 4  # Crisis situations requiring immediate intervention


class TopicMatch(BaseModel):
    """A matched topic with confidence score"""

    topic_id: str = Field(..., description="Unique topic identifier")
    topic_name: str = Field(..., description="Human-readable topic name")
    tier: int = Field(..., description="Safety tier (1-4)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    labels_matched: List[str] = Field(
        default_factory=list, description="Labels that triggered this match"
    )
    hierarchy: Optional[str] = Field(None, description="Topic hierarchy path")


class ClassificationRequest(BaseModel):
    """Request to classify a message"""

    message: str = Field(
        ..., min_length=1, max_length=10000, description="Message text to classify"
    )
    teen_id: Optional[str] = Field(
        None, description="Teen ID for personalized classification"
    )
    context: Optional[Dict[str, Any]] = Field(
        None, description="Additional context for classification"
    )


class ClassificationResponse(BaseModel):
    """Response from topic classification"""

    model_config = ConfigDict(protected_namespaces=())

    message: str = Field(..., description="Original message")
    tier: int = Field(..., description="Highest tier detected (1-4)")
    tier_name: str = Field(..., description="Tier name (GREEN, YELLOW, ORANGE, RED)")
    topics: List[TopicMatch] = Field(
        default_factory=list, description="Detected topics"
    )
    action: str = Field(
        ..., description="Recommended action (allow, require_approval, alert, block)"
    )
    model_used: str = Field(..., description="AI model used for classification")
    processing_time_ms: float = Field(
        ..., description="Processing time in milliseconds"
    )
    cached: bool = Field(
        default=False, description="Whether result was served from cache"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Classification timestamp"
    )


class TaxonomyTopic(BaseModel):
    """Topic in the taxonomy"""

    id: str
    name: str
    tier: int
    level: int
    parent_ids: List[str]
    labels: List[str]
    description: str
    emoji: str
    dashboard_color: str


class TaxonomyResponse(BaseModel):
    """Complete topic taxonomy"""

    total_topics: int
    total_labels: int
    tiers: Dict[int, int] = Field(..., description="Count of topics per tier")
    topics: List[TaxonomyTopic]


class HealthResponse(BaseModel):
    """Health check response"""

    model_config = ConfigDict(protected_namespaces=())

    status: str
    service: str
    version: str
    gemini_model: str
    redis_connected: bool
    classifier_status: Optional[str] = Field(
        default=None, description="Classifier initialization status: 'initializing' | 'ready' | 'failed'"
    )
    model_loaded: bool = Field(
        default=False, description="Whether classifier model is loaded and ready"
    )
    model_path: Optional[str] = Field(
        default=None, description="Path to loaded model, if available"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)
