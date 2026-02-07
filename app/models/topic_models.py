"""Data models for Topic Classifier Service."""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class AgeGroup(str, Enum):
    """Age groups for ModAI users."""
    AGE_13_14 = "13-14"
    AGE_15_16 = "15-16"
    AGE_17_19 = "17-19"


class TopicTier(int, Enum):
    """4-tier topic classification system."""
    GREEN = 1   # Always allowed
    YELLOW = 2  # Needs approval
    ORANGE = 3  # Requires supervision
    RED = 4     # Auto-blocked


class TopicCategory(str, Enum):
    """Topic categories for classification."""
    # Tier 1 (Green) - Always Allowed
    HOMEWORK = "homework"
    HOBBIES = "hobbies"
    GENERAL_KNOWLEDGE = "general_knowledge"
    CAREER_BASIC = "career_basic"
    STUDY_TIPS = "study_tips"
    FRIENDSHIP_BASIC = "friendship_basic"
    TIME_MANAGEMENT = "time_management"
    CREATIVE_WRITING = "creative_writing"

    # Tier 2 (Yellow) - Needs Approval
    SOCIAL_MEDIA = "social_media"
    PEER_PRESSURE = "peer_pressure"
    BODY_IMAGE = "body_image"
    DATING_RELATIONSHIPS = "dating_relationships"
    NEWS_CURRENT_EVENTS = "news_current_events"
    FINANCIAL_LITERACY = "financial_literacy"

    # Tier 3 (Orange) - Requires Supervision
    FAMILY_CONFLICTS = "family_conflicts"
    BULLYING = "bullying"
    STRESS_ANXIETY = "stress_anxiety"
    IDENTITY_QUESTIONS = "identity_questions"
    PUBERTY = "puberty"
    MENTAL_HEALTH = "mental_health"

    # Tier 4 (Red) - Auto-Blocked
    SELF_HARM = "self_harm"
    VIOLENCE = "violence"
    ILLEGAL_ACTIVITIES = "illegal_activities"
    EXPLICIT_SEXUAL = "explicit_sexual"
    HATE_SPEECH = "hate_speech"
    DANGEROUS_CHALLENGES = "dangerous_challenges"


class TopicClassification(BaseModel):
    """Classification result from AI."""
    tier: TopicTier = Field(..., description="Safety tier (1-4)")
    categories: List[TopicCategory] = Field(..., description="Detected topic categories")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    reasoning: str = Field(..., description="Explanation for classification")
    requires_approval: bool = Field(..., description="Whether parental approval is needed")
    crisis_detected: bool = Field(False, description="Whether crisis keywords detected")


class ClassificationRequest(BaseModel):
    """Request model for topic classification."""
    message: str = Field(..., min_length=1, description="Message to classify")
    age_group: AgeGroup = Field(..., description="User's age group")
    conversation_history: Optional[List[str]] = Field(None, description="Recent conversation for context")
    user_id: Optional[str] = Field(None, description="User ID for caching")


class ClassificationResponse(BaseModel):
    """Response model for topic classification."""
    classification: TopicClassification
    cached: bool = Field(False, description="Whether result came from cache")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
