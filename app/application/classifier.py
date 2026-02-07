"""
Topic Classifier using Zero-Shot Classification
"""

import sys
import os
import time
import json
import hashlib
import asyncio
from typing import List, Dict, Any, Optional
import redis
from transformers import pipeline, AutoTokenizer
from fastapi import HTTPException

# Add models directory to path for TOPIC_TAXONOMY import
# Use absolute path in container
# Add models directory to path for TOPIC_TAXONOMY import
# Use absolute path in container, fallback to local relative path
models_path = "/app/models"
if not os.path.exists(models_path):
    models_path = os.path.join(os.getcwd(), "models")

if models_path not in sys.path:
    sys.path.insert(0, models_path)

try:
    from TOPIC_TAXONOMY import (
        TOPICS,
        ALL_LABELS,
        get_topic_by_id,
        classify_by_labels,
        get_tier_topics,
        Tier,
    )
except ImportError as e:
    print(f"Warning: Could not import TOPIC_TAXONOMY: {e}")
    TOPICS = []
    ALL_LABELS = set()

from ..config import settings
from ..domain.models import TopicMatch, ClassificationResponse, Tier as TierEnum


class TopicClassifier:
    """
    Topic classifier that calls the topic-model-server for inference.
    No longer loads models locally - uses HTTP to shared model server.
    """

    def __init__(self):
        """Initialize the classifier - now just sets up HTTP client"""
        print("🔵 TopicClassifier.__init__() START")

        # Set classification method
        self.method = settings.CLASSIFICATION_METHOD
        print(f"🔵 Classification method: {self.method}")

        # Model server URL for ONNX inference
        self.model_server_url = settings.MODEL_SERVER_URL
        print(f"🔵 Model server URL: {self.model_server_url}")

        # For Claude, we still need the API client
        if self.method == "claude":
            print(f"📦 Using Claude API: {settings.CLAUDE_MODEL}...")
            try:
                import anthropic
                self.claude_client = anthropic.Anthropic(
                    api_key=settings.ANTHROPIC_API_KEY
                )
                print(f"✅ Claude API client initialized successfully")
            except ImportError:
                print(f"⚠️ anthropic package not installed, falling back to model server")
                self.method = "onnx"
            except Exception as e:
                print(f"⚠️ Claude API initialization failed: {e}, falling back to model server")
                self.method = "onnx"

        if self.method in ["onnx", "bart"]:
            # For ONNX/BART, we use the model server instead of loading locally
            print(f"📡 Using model server at {self.model_server_url} for inference")
            print("   (Model is loaded in topic-model-server, not here)")

        # Initialize Redis cache
        self.redis_client = None
        if settings.ENABLE_CACHING:
            try:
                self.redis_client = redis.Redis(
                    host=settings.REDIS_HOST,
                    port=settings.REDIS_PORT,
                    db=settings.REDIS_DB,
                    decode_responses=True,
                )
                self.redis_client.ping()
                print(f"✅ Redis connected for caching")
            except Exception as e:
                print(f"⚠️ Redis connection failed: {e}")
                self.redis_client = None

        print(f"✅ Topic Classifier initialized with {self.method} method")
        print(f"✅ {len(TOPICS)} topics, {len(ALL_LABELS)} labels loaded")

    def _get_cache_key(self, message: str) -> str:
        """Generate cache key for message"""
        return f"topic_classifier:{hashlib.md5(message.encode()).hexdigest()}"

    def _get_cached_result(self, message: str) -> Optional[Dict[str, Any]]:
        """Try to get cached classification result"""
        if not self.redis_client:
            return None

        try:
            cache_key = self._get_cache_key(message)
            cached = self.redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            print(f"Cache read error: {e}")

        return None

    def _cache_result(self, message: str, result: Dict[str, Any]):
        """Cache classification result"""
        if not self.redis_client:
            return

        try:
            cache_key = self._get_cache_key(message)
            self.redis_client.setex(
                cache_key, settings.REDIS_CACHE_TTL, json.dumps(result)
            )
        except Exception as e:
            print(f"Cache write error: {e}")

    def _check_crisis_keywords(self, message: str) -> Optional[ClassificationResponse]:
        """
        Check for explicit crisis keywords that should trigger Tier 4 immediately.
        Returns a Tier 4 response if crisis keywords detected, None otherwise.
        """
        message_lower = message.lower()

        # Crisis keyword patterns
        crisis_patterns = [
            # Self-harm
            "hurt myself",
            "harm myself",
            "cut myself",
            "cutting myself",
            # Suicide
            "kill myself",
            "end my life",
            "want to die",
            "suicide",
            "suicidal",
            "don't want to live",
            "better off dead",
            # Violence
            "kill someone",
            "hurt someone",
            "shoot up",
            "bring a gun",
            # Abuse
            "being abused",
            "molested",
            "raped",
            "sexual abuse",
        ]

        detected_patterns = []
        for pattern in crisis_patterns:
            if pattern in message_lower:
                detected_patterns.append(pattern)

        if detected_patterns:
            # Crisis detected - return Tier 4 immediately
            # Find relevant crisis topics
            crisis_topics = []
            for topic in TOPICS:
                if topic.tier.value == 4:
                    # Build hierarchy
                    hierarchy = topic.name
                    if hasattr(topic, "parent_ids") and topic.parent_ids:
                        parent_id = topic.parent_ids[0]
                        parent_topic = get_topic_by_id(parent_id)
                        if parent_topic:
                            hierarchy = f"{parent_topic.name} > {topic.name}"

                    match = TopicMatch(
                        topic_id=topic.id,
                        topic_name=topic.name,
                        tier=4,
                        confidence=1.0,  # High confidence for explicit keywords
                        labels_matched=detected_patterns[
                            :3
                        ],  # Show which patterns matched
                        hierarchy=hierarchy,
                    )
                    crisis_topics.append(match)

            return ClassificationResponse(
                message=message,
                tier=4,
                tier_name="RED",
                topics=crisis_topics[:5],  # Limit to 5 topics
                action="block",
                model_used="crisis-keyword-detection",
                processing_time_ms=0.0,  # Instant detection
                cached=False,
            )

        return None

    async def classify(
        self, message: str, teen_id: Optional[str] = None
    ) -> ClassificationResponse:
        """
        Classify a message into topics

        Args:
            message: The message text to classify
            teen_id: Optional teen ID for personalized classification

        Returns:
            ClassificationResponse with detected topics and recommended action
        """
        start_time = time.time()

        # CRITICAL SAFETY CHECK: Check for crisis keywords FIRST
        crisis_result = self._check_crisis_keywords(message)
        if crisis_result:
            crisis_result.processing_time_ms = (time.time() - start_time) * 1000
            return crisis_result

        # Check cache first
        cached_result = self._get_cached_result(message)
        if cached_result:
            cached_result["cached"] = True
            cached_result["processing_time_ms"] = (time.time() - start_time) * 1000
            return ClassificationResponse(**cached_result)

        # Route to appropriate classification method
        if self.method == "claude":
            return await self._classify_claude(message, start_time)
        elif self.method in ["onnx", "bart"]:
            # Use model server for ONNX/BART
            return await self._classify_zero_shot(message, start_time)
        else:
            # Fallback (shouldn't happen)
            return self._safe_default(message, start_time, "error")

    async def _classify_zero_shot(
        self, message: str, start_time: float
    ) -> ClassificationResponse:
        """Zero-shot classification using model server via HTTP"""
        try:
            import httpx
            
            # Get all labels from taxonomy
            all_labels = list(ALL_LABELS)[:50]  # Limit to 50 most common labels

            # Call model server
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.model_server_url}/predict",
                    json={
                        "text": message,
                        "labels": all_labels,
                        "multi_label": True,
                    },
                )
                response.raise_for_status()
                result = response.json()

            # Get labels above threshold
            detected_labels = [
                label
                for label, score in zip(result["labels"], result["scores"])
                if score >= settings.CONFIDENCE_THRESHOLD
            ]

            # Convert detected labels to topics
            topics = self._labels_to_topics(detected_labels)

            # Determine highest tier
            tier = max([t.tier for t in topics]) if topics else 1
            tier_name = {1: "GREEN", 2: "YELLOW", 3: "ORANGE", 4: "RED"}[tier]

            # Determine action based on tier
            action = self._determine_action(tier)

            # Build response
            processing_time = (time.time() - start_time) * 1000

            model_name = f"model-server-{self.method}"

            result_dict = {
                "message": message,
                "tier": tier,
                "tier_name": tier_name,
                "topics": [t.model_dump() for t in topics],
                "action": action,
                "model_used": model_name,
                "processing_time_ms": processing_time,
                "cached": False,
            }

            # Cache result
            self._cache_result(message, result_dict)

            return ClassificationResponse(**result_dict)

        except Exception as e:
            print(f"Zero-shot classification error: {e}")
            # Return safe default
            return self._safe_default(message, start_time, self.method)

    async def _classify_claude(
        self, message: str, start_time: float
    ) -> ClassificationResponse:
        """Claude API classification with structured output"""
        try:
            # Build topic list for Claude
            topic_descriptions = []
            for topic in TOPICS[:20]:  # Use top 20 topics for prompt brevity
                labels_list = list(topic.labels)[
                    :5
                ]  # Convert set to list, take first 5
                topic_descriptions.append(
                    f"- {topic.name} (Tier {topic.tier.value}): {', '.join(labels_list)}"
                )

            prompt = f"""Classify this teen message into our 4-tier safety system.

Message: "{message}"

Available Topics (59 total, showing top 20):
{chr(10).join(topic_descriptions)}

Safety Tiers:
- Tier 1 (GREEN): Academic, hobbies, general safe topics - always allowed
- Tier 2 (YELLOW): Dating, social media, body image - needs approval
- Tier 3 (ORANGE): Mental health, anxiety, depression - requires supervision
- Tier 4 (RED): Self-harm, suicide, violence - blocked immediately

Respond ONLY with valid JSON in this exact format:
{{"tier": 1, "tier_name": "GREEN", "topics": ["Mathematics", "Homework Help"], "action": "allow", "confidence": 0.95}}

Do not include any explanation, just the JSON object."""

            # Call Claude API
            response = self.claude_client.messages.create(
                model=settings.CLAUDE_MODEL,
                max_tokens=settings.CLAUDE_MAX_TOKENS,
                temperature=settings.CLAUDE_TEMPERATURE,
                messages=[{"role": "user", "content": prompt}],
            )

            # Parse JSON response
            import json

            response_text = response.content[0].text
            result = json.loads(response_text)

            # Map detected topic names to our taxonomy
            detected_topic_names = result.get("topics", [])
            topics = []
            for topic_name in detected_topic_names:
                # Find matching topic in taxonomy
                for topic in TOPICS:
                    if topic.name.lower() == topic_name.lower():
                        # Build hierarchy
                        hierarchy = topic.name
                        if hasattr(topic, "parent_ids") and topic.parent_ids:
                            parent_id = topic.parent_ids[0]
                            parent_topic = get_topic_by_id(parent_id)
                            if parent_topic:
                                hierarchy = f"{parent_topic.name} > {topic.name}"

                        match = TopicMatch(
                            topic_id=topic.id,
                            topic_name=topic.name,
                            tier=topic.tier.value,
                            confidence=result.get("confidence", 0.9),
                            labels_matched=list(topic.labels)[
                                :3
                            ],  # Sample labels (convert set to list)
                            hierarchy=hierarchy,
                        )
                        topics.append(match)
                        break

            tier = result.get("tier", 1)
            tier_name = result.get("tier_name", "GREEN")
            action = result.get("action", "allow")
            processing_time = (time.time() - start_time) * 1000

            result_dict = {
                "message": message,
                "tier": tier,
                "tier_name": tier_name,
                "topics": [t.model_dump() for t in topics],
                "action": action,
                "model_used": f"claude-{settings.CLAUDE_MODEL}",
                "processing_time_ms": processing_time,
                "cached": False,
            }

            # Cache result
            self._cache_result(message, result_dict)

            return ClassificationResponse(**result_dict)

        except Exception as e:
            print(f"Claude classification error: {e}")
            import traceback

            traceback.print_exc()
            # Return safe default
            return self._safe_default(message, start_time, "claude")

    def _safe_default(
        self, message: str, start_time: float, model_used: str
    ) -> ClassificationResponse:
        """Return safe default classification"""
        return ClassificationResponse(
            message=message,
            tier=1,
            tier_name="GREEN",
            topics=[],
            action="allow",
            model_used=model_used,
            processing_time_ms=(time.time() - start_time) * 1000,
            cached=False,
        )

    def _labels_to_topics(self, labels: List[str]) -> List[TopicMatch]:
        """Convert detected labels to topic matches"""
        topic_matches = {}

        for label in labels:
            label_lower = label.lower()

            # Find topics that contain this label
            for topic in TOPICS:
                topic_labels_lower = [l.lower() for l in topic.labels]
                if label_lower in topic_labels_lower:
                    if topic.id not in topic_matches:
                        topic_matches[topic.id] = {
                            "topic": topic,
                            "labels": [],
                            "confidence": 0.0,
                        }
                    topic_matches[topic.id]["labels"].append(label)
                    # Increase confidence for each matched label
                    topic_matches[topic.id]["confidence"] = min(
                        1.0, topic_matches[topic.id]["confidence"] + 0.3
                    )

        # Convert to TopicMatch objects
        matches = []
        for topic_id, data in topic_matches.items():
            topic = data["topic"]

            # Build hierarchy path
            hierarchy = topic.name
            if hasattr(topic, "parent_ids") and topic.parent_ids:
                parent_id = topic.parent_ids[0]
                parent_topic = get_topic_by_id(parent_id)
                if parent_topic:
                    hierarchy = f"{parent_topic.name} > {topic.name}"

            match = TopicMatch(
                topic_id=topic.id,
                topic_name=topic.name,
                tier=topic.tier.value,
                confidence=data["confidence"],
                labels_matched=data["labels"],
                hierarchy=hierarchy,
            )
            matches.append(match)

        # Sort by tier (highest first), then confidence
        matches.sort(key=lambda m: (m.tier, m.confidence), reverse=True)

        # Limit to max topics
        return matches[: settings.MAX_TOPICS_PER_MESSAGE]

    def _determine_action(self, tier: int) -> str:
        """Determine recommended action based on tier"""
        actions = {
            1: "allow",  # Green - allow by default
            2: "require_approval",  # Yellow - require supervisor approval
            3: "alert",  # Orange - alert supervisor, monitor
            4: "block",  # Red - block immediately, alert crisis response
        }
        return actions.get(tier, "allow")

    def get_taxonomy(self) -> Dict[str, Any]:
        """Get the complete topic taxonomy"""
        tier_counts = {}
        for tier in [1, 2, 3, 4]:
            tier_counts[tier] = len(get_tier_topics(tier))

        return {
            "total_topics": len(TOPICS),
            "total_labels": len(ALL_LABELS),
            "tiers": tier_counts,
            "topics": [
                {
                    "id": t.id,
                    "name": t.name,
                    "tier": t.tier.value,
                    "level": t.level,
                    "parent_ids": t.parent_ids,
                    "labels": list(t.labels),
                    "description": t.description,
                    "emoji": t.emoji,
                    "dashboard_color": t.dashboard_color,
                }
                for t in TOPICS
            ],
        }


# Global classifier instance with async initialization
_classifier = None
_classifier_ready = asyncio.Event()
_classifier_error = None
_init_lock = asyncio.Lock()


async def initialize_classifier():
    """
    Initialize classifier in background thread.
    This runs the synchronous TopicClassifier.__init__() in a thread pool
    to avoid blocking the asyncio event loop during ONNX model loading.
    """
    global _classifier, _classifier_error

    async with _init_lock:
        if _classifier is not None:
            return  # Already initialized

        try:
            print("🔄 Initializing classifier in background thread...")
            # Run blocking TopicClassifier() init in thread pool
            loop = asyncio.get_event_loop()
            _classifier = await loop.run_in_executor(
                None,  # Use default ThreadPoolExecutor
                TopicClassifier  # Calls TopicClassifier.__init__()
            )
            _classifier_ready.set()
            print("✅ Classifier initialized successfully")
        except Exception as e:
            _classifier_error = str(e)
            _classifier_ready.set()  # Signal completion even on error
            print(f"❌ Classifier initialization failed: {e}")
            # Don't raise - allow service to continue running with failed classifier


async def get_classifier(timeout: float = 300.0) -> TopicClassifier:
    """
    Get classifier instance, waiting for initialization if needed.

    Args:
        timeout: Maximum seconds to wait for initialization (default 300s = 5min)

    Returns:
        Initialized TopicClassifier instance

    Raises:
        HTTPException: If initialization times out, fails, or classifier is unavailable
    """
    global _classifier, _classifier_error

    # Wait for initialization to complete (or timeout)
    try:
        await asyncio.wait_for(_classifier_ready.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=503,
            detail=f"Classifier still initializing after {timeout}s. Please try again."
        )

    # Check if initialization failed
    if _classifier_error:
        raise HTTPException(
            status_code=500,
            detail=f"Classifier initialization failed: {_classifier_error}"
        )

    # Final safety check
    if _classifier is None:
        raise HTTPException(
            status_code=500,
            detail="Classifier not initialized (unexpected state)"
        )

    return _classifier
