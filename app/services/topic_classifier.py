"""Topic classification service using GPT-4o-mini."""

import json
import hashlib
import logging
from typing import Optional
from openai import AsyncOpenAI
import redis.asyncio as redis

from app.models.topic_models import (
    AgeGroup,
    TopicTier,
    TopicCategory,
    TopicClassification
)
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# Crisis keywords for immediate detection
CRISIS_KEYWORDS = [
    "want to die", "kill myself", "suicide", "self-harm", "self harm",
    "cutting", "hurt myself", "end it all", "not worth living",
    "better off dead", "take my life"
]


class TopicClassifierService:
    """Service for classifying conversation topics using GPT-4o-mini."""

    def __init__(self):
        """Initialize topic classifier."""
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.redis_client = None
        logger.info("Topic Classifier Service initialized")

    async def init_redis(self):
        """Initialize Redis connection."""
        if not self.redis_client:
            self.redis_client = await redis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True
            )

    async def close(self):
        """Close connections."""
        if self.redis_client:
            await self.redis_client.close()

    def _detect_crisis(self, message: str) -> bool:
        """Quick crisis keyword detection."""
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in CRISIS_KEYWORDS)

    def _get_cache_key(self, message: str, age_group: AgeGroup) -> str:
        """Generate cache key for classification."""
        content = f"{message}:{age_group.value}"
        return f"topic_classification:{hashlib.md5(content.encode()).hexdigest()}"

    async def _get_cached_classification(
        self,
        message: str,
        age_group: AgeGroup
    ) -> Optional[TopicClassification]:
        """Try to get classification from cache."""
        try:
            await self.init_redis()
            cache_key = self._get_cache_key(message, age_group)
            cached = await self.redis_client.get(cache_key)
            if cached:
                data = json.loads(cached)
                return TopicClassification(**data)
        except Exception as e:
            logger.warning(f"Cache retrieval error: {e}")
        return None

    async def _cache_classification(
        self,
        message: str,
        age_group: AgeGroup,
        classification: TopicClassification
    ):
        """Cache classification result."""
        try:
            await self.init_redis()
            cache_key = self._get_cache_key(message, age_group)
            await self.redis_client.setex(
                cache_key,
                settings.CACHE_TTL,
                classification.model_dump_json()
            )
        except Exception as e:
            logger.warning(f"Cache write error: {e}")

    def _get_system_prompt(self, age_group: AgeGroup) -> str:
        """Get system prompt for classification based on age group."""
        return f"""You are a topic classifier for a teen safety system (age group: {age_group.value}).

Classify messages into one of 4 safety tiers:

**Tier 1 (GREEN)**: Always safe for teens
- Homework, hobbies, general knowledge, study tips, creative writing, basic career/friendship topics

**Tier 2 (YELLOW)**: Needs parental approval
- Social media advice, peer pressure, body image, dating/relationships, news/politics, financial literacy

**Tier 3 (ORANGE)**: Requires active supervision
- Family conflicts, bullying, stress/anxiety, identity questions, puberty, mental health

**Tier 4 (RED)**: Auto-blocked + crisis resources
- Self-harm, suicide, violence, illegal activities, explicit sexual content, hate speech, dangerous challenges

AGE-SPECIFIC RULES:
- Age 13-14: Dating → Tier 2, Mental Health → Tier 3, Sexual Health → Tier 4
- Age 15-16: Dating → Tier 1, Mental Health → Tier 2, Sexual Health → Tier 3
- Age 17-19: Dating → Tier 1, Mental Health → Tier 2, Sexual Health → Tier 2

Return JSON with:
- tier: 1-4
- categories: list of relevant topic categories
- confidence: 0.0-1.0
- reasoning: brief explanation
- requires_approval: boolean (Tier 2+)
- crisis_detected: boolean (Tier 4 with crisis keywords)

Be cautious - when in doubt, classify to a higher (safer) tier."""

    async def classify(
        self,
        message: str,
        age_group: AgeGroup,
        conversation_history: Optional[list[str]] = None
    ) -> TopicClassification:
        """
        Classify message topic and safety tier.

        Args:
            message: Message to classify
            age_group: User's age group
            conversation_history: Optional recent conversation context

        Returns:
            TopicClassification with tier, categories, and metadata
        """
        # Check cache first
        cached = await self._get_cached_classification(message, age_group)
        if cached:
            logger.info(f"Cache hit for message classification")
            return cached

        # Quick crisis detection
        crisis_detected = self._detect_crisis(message)

        # Build context
        context = message
        if conversation_history:
            context = "Recent conversation:\n" + "\n".join(conversation_history[-3:]) + f"\n\nNew message: {message}"

        try:
            # Call GPT-4o-mini with structured output
            response = await self.client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                temperature=settings.OPENAI_TEMPERATURE,
                messages=[
                    {"role": "system", "content": self._get_system_prompt(age_group)},
                    {"role": "user", "content": context}
                ],
                response_format={"type": "json_object"}
            )

            # Parse response
            result = json.loads(response.choices[0].message.content)

            classification = TopicClassification(
                tier=TopicTier(result["tier"]),
                categories=[TopicCategory(cat) for cat in result["categories"]],
                confidence=result["confidence"],
                reasoning=result["reasoning"],
                requires_approval=result.get("requires_approval", result["tier"] >= 2),
                crisis_detected=crisis_detected or result.get("crisis_detected", False)
            )

            # Cache result
            await self._cache_classification(message, age_group, classification)

            logger.info(f"Classified message as Tier {classification.tier.value} "
                       f"with confidence {classification.confidence:.2f}")

            return classification

        except Exception as e:
            logger.error(f"Classification error: {e}")
            # Fail-safe: classify as Tier 3 (requires supervision)
            return TopicClassification(
                tier=TopicTier.ORANGE,
                categories=[TopicCategory.GENERAL_KNOWLEDGE],
                confidence=0.0,
                reasoning=f"Classification failed: {str(e)}",
                requires_approval=True,
                crisis_detected=crisis_detected
            )


# Singleton instance
_classifier_service = None


async def get_classifier_service() -> TopicClassifierService:
    """Get or create topic classifier service instance."""
    global _classifier_service
    if _classifier_service is None:
        _classifier_service = TopicClassifierService()
    return _classifier_service
