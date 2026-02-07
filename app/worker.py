"""
Background Worker for Topic Classification
Consumes messages from Redis queue and classifies them
"""

import asyncio
import json
import logging
from typing import Optional
from uuid import UUID
import redis.asyncio as redis

from app.application.classifier import get_classifier
from app.config import settings
from app.infrastructure.database import TopicClassifierDatabase

logging.basicConfig(
    level=getattr(
        logging, settings.LOG_LEVEL if hasattr(settings, "LOG_LEVEL") else "INFO"
    ),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class TopicClassificationWorker:
    """Background worker that processes topic classification jobs from Redis queue"""

    QUEUE_NAME = "topic_classification_queue"

    def __init__(
        self,
        redis_url: Optional[str] = None,
        database_url: Optional[str] = None,
    ):
        # If redis_url isn't passed explicitly, build it from settings
        if redis_url:
            self.redis_url = redis_url
        else:
            self.redis_url = f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.REDIS_DB}"
        self.redis_client: Optional[redis.Redis] = None
        self.database = TopicClassifierDatabase(
            database_url
            or getattr(
                settings,
                "DATABASE_URL",
                "postgresql://modai:modai@localhost:5432/modai",
            )
        )
        self.running = False

    async def connect(self):
        """Connect to Redis and Database"""
        if not self.redis_client:
            self.redis_client = await redis.from_url(
                self.redis_url, decode_responses=True
            )
            logger.info("✅ Worker connected to Redis")

        await self.database.connect()

    async def disconnect(self):
        """Disconnect from Redis and Database"""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("Disconnected from Redis")

        await self.database.disconnect()

    async def process_job(self, job_data: dict):
        """
        Process a single classification job

        Args:
            job_data: Job containing message_id, teen_id, content, etc.
        """
        message_id = job_data.get("message_id")
        teen_id = job_data.get("teen_id")
        content = job_data.get("content")

        logger.info(f"📝 Processing classification job for message {message_id}")

        try:
            # Get classifier instance (now async)
            classifier = await get_classifier()

            # Classify the message
            result = await classifier.classify(content, teen_id=teen_id)

            logger.info(
                f"✅ Classified message {message_id}: "
                f"Tier {result.tier} ({result.tier_name}), "
                f"{len(result.topics)} topics, "
                f"Action: {result.action}"
            )

            # Log top topics
            for topic in result.topics[:3]:
                logger.info(
                    f"  - {topic.topic_name} (Tier {topic.tier}, "
                    f"confidence: {topic.confidence:.2f})"
                )

            # 1. Write classification results to message_topics table
            topics_data = [
                {
                    "topic_id": topic.topic_id,
                    "topic_name": topic.topic_name,
                    "tier": topic.tier,
                    "confidence": topic.confidence,
                    "labels_matched": topic.labels_matched,
                    "hierarchy": topic.hierarchy,
                }
                for topic in result.topics
            ]

            await self.database.insert_message_topics(
                message_id=UUID(message_id),
                topics=topics_data,
                model_used=result.model_used,
                processing_time_ms=result.processing_time_ms,
            )

            # 2. Get supervisor for teen (needed for approvals/alerts)
            supervisor_id = await self.database.get_supervisor_for_teen(UUID(teen_id))

            if not supervisor_id:
                logger.warning(
                    f"⚠️ No supervisor found for teen {teen_id}, "
                    f"skipping approval/alert creation"
                )
                return result

            # 3. Handle Tier 2: Create approval request
            if result.action == "require_approval":
                for topic in result.topics:
                    if topic.tier == 2:
                        await self.database.create_approval_request(
                            teen_id=UUID(teen_id),
                            supervisor_id=supervisor_id,
                            message_id=UUID(message_id),
                            topic_id=topic.topic_id,
                            topic_name=topic.topic_name,
                            message_preview=content[:200],
                        )
                        logger.warning(
                            f"⚠️ Tier 2 topic '{topic.topic_name}' - "
                            f"created approval request"
                        )

            # 4. Handle Tier 3: Create alert (mental health)
            elif result.action == "alert":
                for topic in result.topics:
                    if topic.tier == 3:
                        await self.database.create_supervisor_alert(
                            teen_id=UUID(teen_id),
                            supervisor_id=supervisor_id,
                            message_id=UUID(message_id),
                            alert_type="tier3_mental_health",
                            severity="medium",
                            tier=3,
                            topic_id=topic.topic_id,
                            topic_name=topic.topic_name,
                            alert_title=f"Mental Health Topic Detected: {topic.topic_name}",
                            alert_message=(
                                f"Your teen discussed {topic.topic_name.lower()} "
                                f"in their conversation. This may indicate they need "
                                f"additional support."
                            ),
                            message_preview=content[:200],
                        )
                        logger.warning(
                            f"🔔 Tier 3 topic '{topic.topic_name}' - "
                            f"created supervisor alert"
                        )

            # 5. Handle Tier 4: Create critical alert (crisis)
            elif result.action == "block":
                for topic in result.topics:
                    if topic.tier == 4:
                        await self.database.create_supervisor_alert(
                            teen_id=UUID(teen_id),
                            supervisor_id=supervisor_id,
                            message_id=UUID(message_id),
                            alert_type="tier4_crisis",
                            severity="critical",
                            tier=4,
                            topic_id=topic.topic_id,
                            topic_name=topic.topic_name,
                            alert_title=f"URGENT: Crisis Topic Detected - {topic.topic_name}",
                            alert_message=(
                                f"Your teen discussed {topic.topic_name.lower()} "
                                f"which indicates they may be in crisis. "
                                f"IMMEDIATE attention is recommended. "
                                f"Please reach out to your teen and consider "
                                f"professional support."
                            ),
                            message_preview=content[:200],
                        )
                        logger.error(
                            f"🚨 Tier 4 CRISIS topic '{topic.topic_name}' - "
                            f"created CRITICAL alert!"
                        )

            return result

        except Exception as e:
            logger.error(f"❌ Error processing job for message {message_id}: {e}")
            # Job will be lost - could implement retry logic here
            return None

    async def start(self):
        """Start the worker to consume from Redis queue"""
        self.running = True
        await self.connect()
        # Pre-initialize classifier to avoid latency on first job
        try:
            logger.info("⏳ Pre-loading classifier in background thread...")
            # get_classifier() is now async and handles its own background loading
            cls = await get_classifier()
            logger.info(
                f"✅ Classifier pre-loaded: method={cls.method}, model_path={getattr(cls, 'model_path', None)}"
            )
        except Exception as e:
            logger.warning(f"⚠️ Could not pre-load classifier in worker: {e}")

        logger.info(f"🚀 Topic Classification Worker started")
        logger.info(f"📡 Listening to queue: {self.QUEUE_NAME}")

        while self.running:
            try:
                # BLPOP blocks until a job is available (timeout: 1 second)
                result = await self.redis_client.blpop(self.QUEUE_NAME, timeout=1)

                if result:
                    queue_name, job_json = result
                    job_data = json.loads(job_json)

                    # Process the job
                    await self.process_job(job_data)

            except asyncio.CancelledError:
                logger.info("Worker cancelled, shutting down...")
                self.running = False
                break
            except Exception as e:
                logger.error(f"Error in worker loop: {e}")
                await asyncio.sleep(1)  # Brief pause before retry

        await self.disconnect()
        logger.info("Worker stopped")

    def stop(self):
        """Stop the worker"""
        self.running = False


async def main():
    """Main entry point for running the worker"""
    worker = TopicClassificationWorker()

    try:
        await worker.start()
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
        worker.stop()


if __name__ == "__main__":
    asyncio.run(main())
