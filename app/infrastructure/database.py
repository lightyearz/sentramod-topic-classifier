"""
Database connection and operations for Topic Classifier Worker
"""

import asyncpg
import json
import logging
from typing import Optional, List
from uuid import UUID
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class TopicClassifierDatabase:
    """Database operations for topic classification results"""

    def __init__(self, database_url: str):
        self.database_url = database_url
        self.pool: Optional[asyncpg.Pool] = None

    async def connect(self):
        """Create database connection pool"""
        if not self.pool:
            # asyncpg expects a DSN with scheme 'postgres' or 'postgresql'
            # Some services provide a SQLAlchemy-style DSN like 'postgresql+asyncpg://'
            # which asyncpg rejects. Normalize that to 'postgresql://'.
            dsn = self.database_url
            if dsn.startswith("postgresql+asyncpg://"):
                dsn = dsn.replace("postgresql+asyncpg://", "postgresql://", 1)

            self.pool = await asyncpg.create_pool(
                dsn,
                min_size=2,
                max_size=10,
            )
            logger.info("✅ Database connection pool created")

    async def disconnect(self):
        """Close database connection pool"""
        if self.pool:
            await self.pool.close()
            logger.info("Disconnected from database")

    async def insert_message_topics(
        self,
        message_id: UUID,
        topics: List[dict],
        model_used: str,
        processing_time_ms: float,
    ) -> int:
        """
        Insert classification results into message_topics table

        Args:
            message_id: UUID of the message
            topics: List of topic matches with confidence scores
            model_used: AI model name (e.g., "gemini-2.0-flash-exp")
            processing_time_ms: Classification latency

        Returns:
            Number of topics inserted
        """
        if not self.pool:
            raise RuntimeError("Database not connected")

        query = """
            INSERT INTO message_topics (
                message_id,
                topic_id,
                topic_name,
                tier,
                confidence,
                labels_matched,
                hierarchy,
                model_used,
                processing_time_ms
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        """

        inserted = 0
        async with self.pool.acquire() as conn:
            # Ensure the table exists (simple schema for local/dev use)
            logger.info("Ensuring message_topics table exists before inserting")
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS message_topics (
                    id SERIAL PRIMARY KEY,
                    message_id UUID NOT NULL,
                    topic_id TEXT NOT NULL,
                    topic_name TEXT NOT NULL,
                    tier INTEGER,
                    confidence FLOAT,
                    labels_matched JSONB,
                    hierarchy TEXT,
                    model_used TEXT,
                    processing_time_ms FLOAT,
                    created_at TIMESTAMPTZ DEFAULT now()
                );
                """
            )
            logger.info(
                "Ensured message_topics table exists (CREATE TABLE IF NOT EXISTS executed)"
            )
            for topic in topics:
                # Ensure labels_matched is a JSON string for insertion
                if "labels_matched" in topic and not isinstance(
                    topic["labels_matched"], str
                ):
                    logger.debug(
                        "Serializing labels_matched for topic %s (type %s)",
                        topic.get("topic_id"),
                        type(topic["labels_matched"]),
                    )
                    labels_matched_val = json.dumps(topic["labels_matched"])
                else:
                    labels_matched_val = topic.get("labels_matched")
                await conn.execute(
                    query,
                    message_id,
                    topic["topic_id"],
                    topic["topic_name"],
                    topic["tier"],
                    topic["confidence"],
                    labels_matched_val,
                    topic.get("hierarchy", ""),
                    model_used,
                    processing_time_ms,
                )
                inserted += 1

        logger.info(f"✅ Inserted {inserted} topics for message {message_id}")
        return inserted

    async def create_approval_request(
        self,
        teen_id: UUID,
        supervisor_id: UUID,
        message_id: UUID,
        topic_id: str,
        topic_name: str,
        message_preview: str,
    ) -> UUID:
        """
        Create a Tier 2 topic approval request

        Args:
            teen_id: Teen's UUID
            supervisor_id: Supervisor's UUID
            message_id: Message that triggered request
            topic_id: Topic requiring approval
            topic_name: Human-readable topic name
            message_preview: First 200 chars of message

        Returns:
            UUID of created approval request
        """
        if not self.pool:
            raise RuntimeError("Database not connected")

        query = """
            INSERT INTO topic_approval_requests (
                teen_id,
                supervisor_id,
                message_id,
                topic_id,
                topic_name,
                tier,
                message_preview,
                status,
                requested_at,
                expires_at
            ) VALUES ($1, $2, $3, $4, $5, 2, $6, 'pending', $7, $8)
            RETURNING id
        """

        now = datetime.now(timezone.utc)
        expires_at = now.replace(hour=now.hour + 24)  # 24 hours from now

        async with self.pool.acquire() as conn:
            request_id = await conn.fetchval(
                query,
                teen_id,
                supervisor_id,
                message_id,
                topic_id,
                topic_name,
                message_preview[:200],  # Limit to 200 chars
                now,
                expires_at,
            )

        logger.info(
            f"✅ Created approval request {request_id} for topic '{topic_name}'"
        )
        return request_id

    async def create_supervisor_alert(
        self,
        teen_id: UUID,
        supervisor_id: UUID,
        message_id: UUID,
        alert_type: str,
        severity: str,
        tier: int,
        topic_id: str,
        topic_name: str,
        alert_title: str,
        alert_message: str,
        message_preview: Optional[str] = None,
    ) -> UUID:
        """
        Create a Tier 3/4 supervisor alert

        Args:
            teen_id: Teen's UUID
            supervisor_id: Supervisor's UUID
            message_id: Message that triggered alert
            alert_type: Type of alert (tier3_mental_health, tier4_crisis, etc.)
            severity: low/medium/high/critical
            tier: Safety tier (3 or 4)
            topic_id: Topic that triggered alert
            topic_name: Human-readable topic name
            alert_title: Short alert headline
            alert_message: Full alert description
            message_preview: Preview of concerning message

        Returns:
            UUID of created alert
        """
        if not self.pool:
            raise RuntimeError("Database not connected")

        query = """
            INSERT INTO supervisor_alerts (
                teen_id,
                supervisor_id,
                message_id,
                alert_type,
                severity,
                tier,
                topic_id,
                topic_name,
                alert_title,
                alert_message,
                message_preview,
                status
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, 'unread')
            RETURNING id
        """

        async with self.pool.acquire() as conn:
            alert_id = await conn.fetchval(
                query,
                teen_id,
                supervisor_id,
                message_id,
                alert_type,
                severity,
                tier,
                topic_id,
                topic_name,
                alert_title,
                alert_message,
                message_preview[:200] if message_preview else None,
            )

        logger.warning(
            f"🚨 Created {severity} alert {alert_id} for topic '{topic_name}' (Tier {tier})"
        )
        return alert_id

    async def get_supervisor_for_teen(self, teen_id: UUID) -> Optional[UUID]:
        """
        Get supervisor ID for a teen from the relationships table

        Args:
            teen_id: Teen's UUID

        Returns:
            Supervisor's UUID or None if not found
        """
        if not self.pool:
            raise RuntimeError("Database not connected")

        query = """
            SELECT supervisor_id
            FROM relationships
            WHERE teen_id = $1
              AND status = 'active'
            LIMIT 1
        """

        async with self.pool.acquire() as conn:
            try:
                supervisor_id = await conn.fetchval(query, teen_id)
                return supervisor_id
            except Exception as e:
                # If the relationships table isn't present in local/dev DB, log and return None
                if 'relation "relationships" does not exist' in str(e):
                    logger.warning(
                        "relationships table not found in DB; get_supervisor_for_teen will return None"
                    )
                    return None
                else:
                    # Re-raise unexpected exceptions so they can be handled by callers
                    raise
