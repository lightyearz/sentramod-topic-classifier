"""
Age Limit and Approval Service for Topic Classifier.

Integrates with User Service to:
- Check daily message limits based on age group
- Check if topic approvals exist
- Create topic approval requests when needed
"""

import logging
import os
from typing import Optional, Dict, Any, List
from uuid import UUID

import httpx

logger = logging.getLogger(__name__)

# User Service URL from environment
USER_SERVICE_URL = os.getenv("USER_SERVICE_URL", "http://localhost:8001")


class AgeLimitService:
    """
    Service for checking age-based limits and topic approvals.

    Integrates with User Service APIs:
    - /api/v1/usage/today - Get today's usage
    - /api/v1/age-groups/for-age/{age} - Get age group config
    - /api/v1/topic-approvals - Check/create approval requests
    """

    def __init__(self, user_service_url: str = USER_SERVICE_URL):
        self.user_service_url = user_service_url
        self.usage_today_url = f"{user_service_url}/api/v1/usage/today"
        self.age_groups_url = f"{user_service_url}/api/v1/age-groups"
        self.topic_approvals_url = f"{user_service_url}/api/v1/topic-approvals"

    async def check_daily_message_limit(
        self,
        teen_id: str,
        teen_age: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Check if teen has reached their daily message limit.

        Args:
            teen_id: Teen's user ID
            teen_age: Teen's age (optional, for age group lookup)

        Returns:
            {
                "allowed": bool,
                "messages_sent": int,
                "messages_limit": int,
                "messages_remaining": int,
                "error": str (if any)
            }
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                # Get today's usage
                response = await client.get(
                    self.usage_today_url,
                    params={"user_id": teen_id},
                )

                if response.status_code == 200:
                    usage_data = response.json()
                    messages_sent = usage_data.get("messages_sent", 0)

                    # Get age group config to check limit
                    if teen_age:
                        age_group_response = await client.get(
                            f"{self.age_groups_url}/for-age/{teen_age}"
                        )

                        if age_group_response.status_code == 200:
                            age_config = age_group_response.json()
                            messages_limit = age_config.get("max_daily_messages", 100)
                        else:
                            # Default fallback
                            messages_limit = 100
                    else:
                        # Default fallback if age not provided
                        messages_limit = 100

                    messages_remaining = max(0, messages_limit - messages_sent)
                    allowed = messages_sent < messages_limit

                    return {
                        "allowed": allowed,
                        "messages_sent": messages_sent,
                        "messages_limit": messages_limit,
                        "messages_remaining": messages_remaining,
                    }
                elif response.status_code == 404:
                    # No usage record yet (first message of the day)
                    return {
                        "allowed": True,
                        "messages_sent": 0,
                        "messages_limit": 100,
                        "messages_remaining": 100,
                    }
                else:
                    logger.warning(
                        f"Usage check failed: status={response.status_code}, teen={teen_id}"
                    )
                    # Fail open (allow message) rather than fail closed
                    return {
                        "allowed": True,
                        "messages_sent": 0,
                        "messages_limit": 100,
                        "messages_remaining": 100,
                        "error": "Usage service unavailable",
                    }

        except httpx.TimeoutException:
            logger.error(f"Usage check timeout for teen={teen_id}")
            # Fail open
            return {
                "allowed": True,
                "messages_sent": 0,
                "messages_limit": 100,
                "messages_remaining": 100,
                "error": "Timeout",
            }
        except Exception as e:
            logger.error(f"Usage check error for teen={teen_id}: {e}")
            # Fail open
            return {
                "allowed": True,
                "messages_sent": 0,
                "messages_limit": 100,
                "messages_remaining": 100,
                "error": str(e),
            }

    async def check_topic_approval(
        self,
        teen_id: str,
        topic_category: str,
        topic_tier: int,
    ) -> Dict[str, Any]:
        """
        Check if topic approval exists for this teen and topic.

        Args:
            teen_id: Teen's user ID
            topic_category: Topic category name
            topic_tier: Topic tier (1-4)

        Returns:
            {
                "approved": bool,
                "requires_approval": bool,
                "approval_request_id": str (if exists),
                "status": str (pending/approved/denied),
                "error": str (if any)
            }
        """
        try:
            # Tier 1 (GREEN) - Always approved
            if topic_tier == 1:
                return {
                    "approved": True,
                    "requires_approval": False,
                }

            # Tier 4 (RED) - Always blocked
            if topic_tier == 4:
                return {
                    "approved": False,
                    "requires_approval": False,  # Not approvable
                }

            # Tier 2/3 (YELLOW/ORANGE) - Check for approval
            async with httpx.AsyncClient(timeout=5.0) as client:
                # Check for existing approval requests
                response = await client.get(
                    f"{self.topic_approvals_url}/pending",
                    params={"teen_id": teen_id},
                )

                if response.status_code == 200:
                    data = response.json()
                    pending_requests = data.get("pending_requests", [])

                    # Check if this specific topic/category has approval
                    for req in pending_requests:
                        if req.get("topic_category") == topic_category:
                            status = req.get("status", "pending")

                            if status == "approved":
                                return {
                                    "approved": True,
                                    "requires_approval": True,
                                    "approval_request_id": req.get("id"),
                                    "status": "approved",
                                }
                            elif status == "denied":
                                return {
                                    "approved": False,
                                    "requires_approval": True,
                                    "approval_request_id": req.get("id"),
                                    "status": "denied",
                                }
                            else:  # pending
                                return {
                                    "approved": False,
                                    "requires_approval": True,
                                    "approval_request_id": req.get("id"),
                                    "status": "pending",
                                }

                    # No approval request exists yet
                    return {
                        "approved": False,
                        "requires_approval": True,
                        "status": "none",
                    }
                else:
                    logger.warning(
                        f"Approval check failed: status={response.status_code}, teen={teen_id}"
                    )
                    # Fail open (allow) for availability
                    return {
                        "approved": True,
                        "requires_approval": True,
                        "error": "Approval service unavailable",
                    }

        except httpx.TimeoutException:
            logger.error(f"Approval check timeout for teen={teen_id}")
            # Fail open
            return {
                "approved": True,
                "requires_approval": True,
                "error": "Timeout",
            }
        except Exception as e:
            logger.error(f"Approval check error for teen={teen_id}: {e}")
            # Fail open
            return {
                "approved": True,
                "requires_approval": True,
                "error": str(e),
            }

    async def create_approval_request(
        self,
        teen_id: str,
        topic_category: str,
        topic_tier: int,
        message_preview: str,
    ) -> Optional[str]:
        """
        Create a new topic approval request.

        Args:
            teen_id: Teen's user ID
            topic_category: Topic category name
            topic_tier: Topic tier (1-4)
            message_preview: First 200 chars of teen's message

        Returns:
            Approval request ID if created, None if failed
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(
                    self.topic_approvals_url,
                    json={
                        "teen_id": teen_id,
                        "topic_category": topic_category,
                        "topic_tier": topic_tier,
                        "message_preview": message_preview[:200],  # Truncate to 200 chars
                    },
                )

                if response.status_code == 201:
                    data = response.json()
                    approval_request = data.get("approval_request", {})
                    approval_id = approval_request.get("id")

                    logger.info(
                        f"📬 Created approval request: id={approval_id}, teen={teen_id}, topic={topic_category}"
                    )
                    return approval_id
                else:
                    logger.warning(
                        f"Failed to create approval request: status={response.status_code}, teen={teen_id}"
                    )
                    return None

        except Exception as e:
            logger.error(f"Error creating approval request for teen={teen_id}: {e}")
            return None

    async def check_availability(
        self,
        teen_id: str,
    ) -> Dict[str, Any]:
        """
        Check if a teen is currently in a blocked (blackout) period.

        Calls User Service time-management endpoint which evaluates all
        supervisor blackout periods against the teen's current time/timezone.

        Returns:
            {"allowed": bool, "reason": str | None}
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(
                    f"{self.user_service_url}/api/v1/time-management/teen/{teen_id}/check-availability"
                )

                if response.status_code == 200:
                    data = response.json()
                    is_blocked = data.get("is_blocked", False)
                    return {
                        "allowed": not is_blocked,
                        "reason": data.get("reason"),
                    }
                else:
                    # Fail open — don't block on service errors
                    logger.warning(
                        f"Availability check failed: status={response.status_code}, teen={teen_id}"
                    )
                    return {"allowed": True, "reason": None}

        except httpx.TimeoutException:
            logger.error(f"Availability check timeout for teen={teen_id}")
            return {"allowed": True, "reason": None}
        except Exception as e:
            logger.error(f"Availability check error for teen={teen_id}: {e}")
            return {"allowed": True, "reason": None}

    async def get_teen_topic_permissions(
        self,
        teen_id: str,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get effective topic permissions for a teen (merged from all supervisors).

        Args:
            teen_id: Teen's user ID

        Returns:
            Dict mapping topic_id to permission details:
            {
                "gaming_video_games": {"enabled": False, "tier_override": 3},
                "mental_health_anxiety": {"enabled": True, "tier_override": None},
            }
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(
                    f"{self.user_service_url}/api/v1/topic-permissions/effective/{teen_id}"
                )

                if response.status_code == 200:
                    data = response.json()
                    # Convert list to dict for easier lookup
                    permissions_dict = {}
                    for perm in data:
                        permissions_dict[perm["topic_id"]] = {
                            "enabled": perm["enabled"],
                            "tier_override": perm.get("effective_tier"),
                        }
                    
                    logger.info(
                        f"📋 Loaded {len(permissions_dict)} topic permissions for teen {teen_id}"
                    )
                    return permissions_dict
                elif response.status_code == 404:
                    # No permissions set yet
                    return {}
                else:
                    logger.warning(
                        f"Permission fetch failed: status={response.status_code}, teen={teen_id}"
                    )
                    # Fail open (no restrictions)
                    return {}

        except httpx.TimeoutException:
            logger.error(f"Permission fetch timeout for teen={teen_id}")
            # Fail open
            return {}
        except Exception as e:
            logger.error(f"Permission fetch error for teen={teen_id}: {e}")
            # Fail open
            return {}


# Singleton instance
_age_limit_service: Optional[AgeLimitService] = None


def get_age_limit_service() -> AgeLimitService:
    """Get or create the age limit service instance."""
    global _age_limit_service
    if _age_limit_service is None:
        _age_limit_service = AgeLimitService()
    return _age_limit_service
