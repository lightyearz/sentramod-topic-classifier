"""
Chat Completions API Routes - OpenAI-Compatible Gateway
This makes the Topic Classifier service act as a safety gateway for chat.
LibreChat calls these endpoints instead of calling AI providers directly.

Flow:
1. Receive chat request from LibreChat
2. Classify user message (Topic Classifier)
3. Block/require approval based on tier
4. Route to AI provider (OpenAI/Claude/Gemini)
5. Classify AI response
6. Persist to Message Service
7. Return response with safety metadata
"""

import logging
import time
import os
from typing import Optional, List, Dict, Any
from uuid import UUID
from fastapi import APIRouter, HTTPException, Depends, Header
from pydantic import BaseModel, Field
import httpx

from ..application.classifier import get_classifier, TopicClassifier
from ..config import settings
from ..infrastructure.age_limit_service import get_age_limit_service

logger = logging.getLogger(__name__)

# Message Service URL
MESSAGE_SERVICE_URL = os.getenv("MESSAGE_SERVICE_URL", "http://localhost:8007")

# Create API router with OpenAI-compatible prefix
chat_router = APIRouter(prefix="/v1", tags=["chat"])


# ============================================
# Pydantic Models (OpenAI-compatible format)
# ============================================

class ChatMessage(BaseModel):
    role: str  # "system", "user", "assistant"
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "modai-safe"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1000
    stream: Optional[bool] = False
    # ModAI-specific fields
    user_age: Optional[int] = 15
    teen_id: Optional[str] = None
    conversation_id: Optional[str] = None  # For message persistence


class SafetyMetadata(BaseModel):
    tier: int
    tier_name: str
    categories: List[str]
    confidence: float
    requires_approval: bool = False
    crisis_detected: bool = False


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage
    # ModAI extensions
    safety: Optional[SafetyMetadata] = None
    blocked: bool = False
    requires_approval: bool = False
    conversation_id: Optional[str] = None  # For continuing conversations


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = 1700000000
    owned_by: str = "modai"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


# ============================================
# AI Provider Routing
# ============================================

# Model to provider mapping
MODEL_PROVIDERS = {
    "modai-safe": "gemini",  # Default routes to Gemini
    "gpt-4o": "openai",
    "gpt-4o-mini": "openai",
    "claude-sonnet-4-5-20250929": "anthropic",
    "claude-3-5-haiku-20241022": "anthropic",
    "gemini-2.0-flash-exp": "gemini",
}


async def call_openai(messages: List[ChatMessage], model: str, temperature: float, max_tokens: int) -> str:
    """Call OpenAI API"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [{"role": m.role, "content": m.content} for m in messages],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]


async def call_anthropic(messages: List[ChatMessage], model: str, temperature: float, max_tokens: int) -> str:
    """Call Anthropic Claude API"""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Anthropic API key not configured")

    # Extract system message
    system_msg = ""
    chat_messages = []
    for m in messages:
        if m.role == "system":
            system_msg = m.content
        else:
            chat_messages.append({"role": m.role, "content": m.content})

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": max_tokens,
                "system": system_msg if system_msg else "You are a helpful AI assistant for teenagers.",
                "messages": chat_messages,
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["content"][0]["text"]


async def call_gemini(messages: List[ChatMessage], model: str, temperature: float, max_tokens: int) -> str:
    """Call Google Gemini API"""
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Google API key not configured")

    # Convert messages to Gemini format
    contents = []
    system_instruction = None

    for m in messages:
        if m.role == "system":
            system_instruction = m.content
        elif m.role == "user":
            contents.append({"role": "user", "parts": [{"text": m.content}]})
        elif m.role == "assistant":
            contents.append({"role": "model", "parts": [{"text": m.content}]})

    async with httpx.AsyncClient(timeout=60.0) as client:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

        body = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }

        if system_instruction:
            body["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        response = await client.post(
            url,
            params={"key": api_key},
            json=body,
        )
        response.raise_for_status()
        data = response.json()

        # Extract text from Gemini response
        candidates = data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            if parts:
                return parts[0].get("text", "")

        return "I apologize, but I couldn't generate a response. Please try again."


async def route_to_ai(
    messages: List[ChatMessage],
    model: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Route request to appropriate AI provider"""
    provider = MODEL_PROVIDERS.get(model, "gemini")

    logger.info(f"🤖 Routing to {provider} (model: {model})")

    if provider == "openai":
        return await call_openai(messages, model, temperature, max_tokens)
    elif provider == "anthropic":
        return await call_anthropic(messages, model, temperature, max_tokens)
    else:  # gemini
        actual_model = "gemini-2.0-flash-exp" if model == "modai-safe" else model
        return await call_gemini(messages, actual_model, temperature, max_tokens)


# ============================================
# Message Service Integration
# ============================================

async def create_conversation(teen_id: str, title: str = "New Conversation") -> Optional[str]:
    """Create a new conversation in Message Service"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{MESSAGE_SERVICE_URL}/api/v1/conversations",
                json={"teen_id": teen_id, "title": title},
            )
            if response.status_code == 201:
                data = response.json()
                return data.get("id")
    except Exception as e:
        logger.error(f"Failed to create conversation: {e}")
    return None


async def save_message(
    conversation_id: str,
    role: str,
    content: str,
    tier: int = 1,
    categories: List[str] = None,
) -> bool:
    """Save a message to Message Service"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{MESSAGE_SERVICE_URL}/api/v1/conversations/{conversation_id}/messages",
                json={
                    "role": role,
                    "content": content,
                    "topic_tier": tier,
                    "topic_categories": categories or [],
                },
            )
            return response.status_code == 201
    except Exception as e:
        logger.error(f"Failed to save message: {e}")
    return False


async def update_conversation_title(conversation_id: str, title: str) -> bool:
    """Update conversation title in Message Service"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.patch(
                f"{MESSAGE_SERVICE_URL}/api/v1/conversations/{conversation_id}",
                json={"title": title},
            )
            return response.status_code == 200
    except Exception as e:
        logger.error(f"Failed to update conversation title: {e}")
    return False


def generate_title_from_message(message: str) -> str:
    """Generate a conversation title from the first message"""
    # Take first 50 chars, find last space, append ...
    if len(message) <= 50:
        return message
    truncated = message[:50]
    last_space = truncated.rfind(" ")
    if last_space > 20:
        return truncated[:last_space] + "..."
    return truncated + "..."


# ============================================
# Age-Appropriate System Prompts
# ============================================

def get_system_prompt_for_age(age: int) -> str:
    """Get age-appropriate system prompt with educational philosophy.
    
    NOTE: This prompt focuses ONLY on teaching approach.
    Topic restrictions are handled by the Topic Classifier, not here.
    """
    base = """You are ModAI, a helpful and supportive AI learning assistant for teenagers.

YOUR CORE PHILOSOPHY - Guide, Don't Give Answers:
- NEVER give direct answers to homework, tests, or assignments
- Instead, guide students to understand the concept themselves
- Ask clarifying questions: "What do you already know about this?"
- Break problems into smaller steps and let THEM solve each step
- Explain the "why" behind concepts, not just the "what"
- When they're stuck, give hints, not solutions
- Celebrate when they figure things out: "You got it!"

Your role is to:
- Help students LEARN, not complete assignments for them
- Provide career and college guidance
- Answer questions thoughtfully and helpfully
- Be encouraging, patient, and understanding
- Use clear language suitable for teens

Communication style:
- Be warm and approachable
- Encourage critical thinking and problem-solving
- Celebrate effort and progress

EXAMPLE APPROACH for homework help:
❌ BAD: "The answer to 2x + 5 = 15 is x = 5"
✅ GOOD: "Let's work through this together! What's the first step to isolate x? What operation would undo the +5?"

❌ BAD: "The theme of the book is..." (giving the answer)
✅ GOOD: "What emotions did you feel while reading? What do you think the author wanted you to take away?"

RESPONSE STRUCTURE for academic help:
When helping with homework, math, science, or any problem-solving:

1. THINKING: Brief acknowledgment of the problem type
   Example: "This is a quadratic equation - let's factor it!"

2. STEPS: Break into numbered steps, guide them through each one
   - Step 1: Identify what we need to find
   - Step 2: What method should we use?
   - Step 3: Apply the method (guide, don't solve for them!)

3. HINTS: If they're stuck, give progressively helpful hints
   - First hint: General direction ("What two numbers multiply to 6?")
   - Second hint: More specific guidance ("Try 2 and 3...")
   - Never jump straight to the answer

4. SOLUTION CHECK: Only confirm when THEY provide the answer
   - If wrong: "Almost! Check step 2 again - what numbers add to 5?"
   - If right: "You got it! 🎉 Great work!"

For NON-ACADEMIC topics (advice, feelings, stress, friendship):
- Use a conversational, supportive tone - no numbered steps needed
- Validate their feelings first ("It's totally normal to feel that way!")
- Offer perspective and suggestions, not rigid solutions
- Be warm and relatable, like a supportive older friend

Remember: Your goal is to create confident, independent learners - not dependent ones."""

    if age <= 14:
        return f"""{base}

Age Group: 13-14 years old
- Use simple, encouraging explanations
- Focus on school fundamentals and hobbies
- Be extra patient - they're still developing study skills
- Celebrate small wins to build confidence
- Use relatable examples from their world
- For STEPS: Keep to 2-3 simple steps max, break down further if needed
- For HINTS: Give more frequent, smaller hints - don't let them get frustrated"""
    elif age <= 16:
        return f"""{base}

Age Group: 15-16 years old
- Discuss more complex academic topics
- Support career exploration with guiding questions
- Balance independence with guidance
- Encourage them to form their own opinions
- Help them develop critical thinking skills
- For STEPS: Can handle 3-4 step problems, push them to think ahead
- For HINTS: Wait a bit longer before giving hints - let them struggle productively"""
    else:
        return f"""{base}

Age Group: 17-19 years old
- Discuss college and career planning
- Support advanced academic topics
- Respect their growing independence
- Treat them more as intellectual equals
- Help them prepare for adult decision-making
- For STEPS: Can handle complex multi-step problems, expect more from them
- For HINTS: Only give hints if they explicitly ask - encourage independent problem-solving"""


# ============================================
# API Endpoints
# ============================================

@chat_router.post("/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(
    request: ChatCompletionRequest,
    classifier: TopicClassifier = Depends(get_classifier),
    authorization: Optional[str] = Header(None),
):
    """
    OpenAI-compatible chat completions endpoint with safety scanning.

    This endpoint:
    1. Classifies the user message for safety (pre-check)
    2. Blocks Tier 4 (RED) messages, returns crisis resources
    3. Flags Tier 2/3 (YELLOW/ORANGE) for approval
    4. Allows Tier 1 (GREEN), routes to AI
    5. Classifies AI response (post-check)
    6. Returns response with safety metadata
    """
    start_time = time.time()

    # Get the last user message
    user_messages = [m for m in request.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message found")

    last_user_message = user_messages[-1].content

    # ==========================================
    # STEP 0: Blocked Hours Enforcement
    # ==========================================
    if request.teen_id:
        age_limit_service = get_age_limit_service()
        availability = await age_limit_service.check_availability(teen_id=request.teen_id)

        if not availability.get("allowed", True):
            logger.warning(
                f"🚫 BLOCKED HOURS: teen={request.teen_id}, reason={availability.get('reason')}"
            )
            return ChatCompletionResponse(
                id=f"chatcmpl-blackout-{int(time.time())}",
                created=int(time.time()),
                model=request.model,
                choices=[ChatCompletionChoice(
                    message=ChatMessage(
                        role="assistant",
                        content=f"It looks like this time is currently set as a break period by your parent/guardian. "
                               f"({availability.get('reason', 'Scheduled blackout')})\n\n"
                               "You'll be able to continue chatting when the break period ends. "
                               "In the meantime, maybe try reading a book, spending time outside, or doing something fun offline! 😊"
                    ),
                    finish_reason="blackout",
                )],
                usage=ChatCompletionUsage(),
                blocked=True,
            )

    # ==========================================
    # STEP 1: Pre-Classification (Safety Check)
    # ==========================================
    logger.info(f"🔍 Classifying user message: {last_user_message[:50]}...")

    try:
        classification = await classifier.classify(
            message=last_user_message,
            teen_id=request.teen_id,
        )
        tier = classification.tier
        tier_name = classification.tier_name
        topics = [t.topic_name for t in classification.topics]
        confidence = classification.topics[0].confidence if classification.topics else 0.9
    except Exception as e:
        logger.error(f"Classification error: {e}")
        # Fail-safe: allow with caution
        tier = 1
        tier_name = "GREEN"
        topics = ["unknown"]
        confidence = 0.5

    safety_metadata = SafetyMetadata(
        tier=tier,
        tier_name=tier_name,
        categories=topics,
        confidence=confidence,
        requires_approval=tier in [2, 3],
        crisis_detected=tier == 4,
    )

    # ==========================================
    # STEP 1.4: Check Topic Permissions (Supervisor Overrides)
    # ==========================================
    if request.teen_id and classification.topics:
        age_limit_service = get_age_limit_service()
        
        # Get effective permissions (merged from all supervisors)
        permissions = await age_limit_service.get_teen_topic_permissions(
            teen_id=request.teen_id
        )
        
        # Check each detected topic against permissions
        for topic in classification.topics:
            topic_id = topic.topic_id
            topic_name = topic.topic_name
            
            if topic_id in permissions:
                perm = permissions[topic_id]
                
                # Check if topic is disabled
                if not perm["enabled"]:
                    logger.warning(
                        f"🚫 TOPIC DISABLED: teen={request.teen_id}, topic={topic_name} (by supervisor)"
                    )
                    return ChatCompletionResponse(
                        id=f"chatcmpl-disabled-{int(time.time())}",
                        created=int(time.time()),
                        model=request.model,
                        choices=[ChatCompletionChoice(
                            message=ChatMessage(
                                role="assistant",
                                content=f"This topic ({topic_name}) has been disabled by your supervisor.\\n\\n"
                                       "If you believe this should be allowed, please talk to your parent/guardian. "
                                       "They can adjust topic permissions in their dashboard.\\n\\n"
                                       "Is there something else I can help you with? 😊"
                            ),
                            finish_reason="topic_disabled",
                        )],
                        usage=ChatCompletionUsage(),
                        safety=safety_metadata,
                        blocked=True,
                    )
                
                # Check tier override (supervisor can increase restriction)
                tier_override = perm.get("tier_override")
                if tier_override and tier_override > tier:
                    logger.warning(
                        f"⬆️ TIER OVERRIDE: teen={request.teen_id}, topic={topic_name}, "
                        f"{tier} -> {tier_override} (by supervisor)"
                    )
                    # Update tier to more restrictive level
                    tier = tier_override
                    tier_name = f"TIER_{tier_override}"
                    # Re-evaluate based on new tier (will be handled below in Steps 2-3)

    # ==========================================
    # STEP 1.5: Check Age Limits & Daily Usage
    # ==========================================
    if request.teen_id:
        age_limit_service = get_age_limit_service()

        # Check daily message limit
        usage_check = await age_limit_service.check_daily_message_limit(
            teen_id=request.teen_id,
            teen_age=request.user_age,
        )

        if not usage_check.get("allowed", True):
            logger.warning(
                f"⏸️ DAILY LIMIT REACHED: teen={request.teen_id}, "
                f"sent={usage_check.get('messages_sent')}/{usage_check.get('messages_limit')}"
            )
            return ChatCompletionResponse(
                id=f"chatcmpl-limit-{int(time.time())}",
                created=int(time.time()),
                model=request.model,
                choices=[ChatCompletionChoice(
                    message=ChatMessage(
                        role="assistant",
                        content=f"You've reached your daily message limit ({usage_check.get('messages_limit')} messages). "
                               "Please take a break and come back tomorrow! 😊\n\n"
                               "Remember: It's healthy to balance screen time with other activities."
                    ),
                    finish_reason="daily_limit",
                )],
                usage=ChatCompletionUsage(),
                safety=safety_metadata,
                blocked=True,
            )

    # ==========================================
    # STEP 1.6: Check Topic Approvals (Tier 3 ONLY)
    # ==========================================
    if tier == 3 and request.teen_id:
        age_limit_service = get_age_limit_service()
        topic_category = topics[0] if topics else "general"

        # Check if approval exists
        approval_check = await age_limit_service.check_topic_approval(
            teen_id=request.teen_id,
            topic_category=topic_category,
            topic_tier=tier,
        )

        if not approval_check.get("approved", False):
            approval_status = approval_check.get("status", "none")

            if approval_status == "none":
                # Create new approval request
                approval_id = await age_limit_service.create_approval_request(
                    teen_id=request.teen_id,
                    topic_category=topic_category,
                    topic_tier=tier,
                    message_preview=last_user_message,
                )

                logger.info(
                    f"📬 APPROVAL REQUESTED: teen={request.teen_id}, topic={topic_category}, tier={tier}, id={approval_id}"
                )

                return ChatCompletionResponse(
                    id=f"chatcmpl-approval-{int(time.time())}",
                    created=int(time.time()),
                    model=request.model,
                    choices=[ChatCompletionChoice(
                        message=ChatMessage(
                            role="assistant",
                            content=f"This topic ({topic_category}) requires parent approval before we can discuss it.\n\n"
                                   "I've sent a request to your parent/guardian. They'll be notified via email and can "
                                   "approve or deny this topic in their dashboard.\n\n"
                                   "You'll be able to discuss this once they approve! In the meantime, is there something else "
                                   "I can help you with? 😊"
                        ),
                        finish_reason="requires_approval",
                    )],
                    usage=ChatCompletionUsage(),
                    safety=safety_metadata,
                    requires_approval=True,
                )

            elif approval_status == "pending":
                # Approval request already exists, waiting for parent
                logger.info(
                    f"⏳ WAITING FOR APPROVAL: teen={request.teen_id}, topic={topic_category}, tier={tier}"
                )

                return ChatCompletionResponse(
                    id=f"chatcmpl-waiting-{int(time.time())}",
                    created=int(time.time()),
                    model=request.model,
                    choices=[ChatCompletionChoice(
                        message=ChatMessage(
                            role="assistant",
                            content=f"We're still waiting for your parent/guardian to approve this topic ({topic_category}).\n\n"
                                   "They've been notified and can review your request in their dashboard. "
                                   "This usually takes a few hours.\n\n"
                                   "Is there something else I can help you with in the meantime? 😊"
                        ),
                        finish_reason="waiting_approval",
                    )],
                    usage=ChatCompletionUsage(),
                    safety=safety_metadata,
                    requires_approval=True,
                )

            elif approval_status == "denied":
                # Parent denied this topic
                logger.info(
                    f"🚫 APPROVAL DENIED: teen={request.teen_id}, topic={topic_category}, tier={tier}"
                )

                return ChatCompletionResponse(
                    id=f"chatcmpl-denied-{int(time.time())}",
                    created=int(time.time()),
                    model=request.model,
                    choices=[ChatCompletionChoice(
                        message=ChatMessage(
                            role="assistant",
                            content=f"Your parent/guardian has decided not to approve discussions about {topic_category} at this time.\n\n"
                                   "If you'd like to talk with them about why, you can discuss it together. "
                                   "They want to make sure you're safe and supported! 💙\n\n"
                                   "Is there something else I can help you with? 😊"
                        ),
                        finish_reason="approval_denied",
                    )],
                    usage=ChatCompletionUsage(),
                    safety=safety_metadata,
                    blocked=True,
                )

    # ==========================================
    # STEP 2: Handle Based on Safety Tier
    # ==========================================

    # TIER 4 (RED) - Block + Alert Supervisor + Crisis Resources
    if tier == 4:
        logger.error(f"🚨 TIER 4 CRISIS: teen={request.teen_id}, topics={topics}")
        
        # Alert supervisor (if teen_id available)
        # This is handled async by the worker, but we log it here
        # The worker will create the alert when processing message_topics
        
        from app.application.response_generator import ResponseGenerator
        crisis_message = ResponseGenerator.get_tier_4_crisis()
        
        return ChatCompletionResponse(
            id=f"chatcmpl-crisis-{int(time.time())}",
            created=int(time.time()),
            model=request.model,
            choices=[ChatCompletionChoice(
                message=ChatMessage(
                    role="assistant",
                    content=crisis_message
                ),
                finish_reason="crisis_detected",
            )],
            usage=ChatCompletionUsage(),
            safety=safety_metadata,
            blocked=True,
        )

    # TIER 2 (YELLOW) - Guidance, conversation continues with context
    if tier == 2:
        logger.info(f"💡 TIER 2 GUIDANCE: teen={request.teen_id}, topic={topics[0] if topics else 'unknown'}")
        
        # Add guidance to system prompt instead of blocking
        from app.application.response_generator import ResponseGenerator
        guidance = ResponseGenerator.get_tier_2_guidance(
            topic_name=topics[0] if topics else "this topic",
            topic_category=topics[0] if topics else None
        )
        
        # Prepend guidance to conversation
        messages_with_guidance = request.messages.copy()
        messages_with_guidance.insert(-1, ChatMessage(
            role="assistant",
            content=guidance
        ))
        
        # Update request messages with guidance
        request.messages = messages_with_guidance
        # Continue to Step 3 (conversation will proceed with AI)

    # ==========================================
    # STEP 3: Prepare Conversation (Message Persistence)
    # ==========================================
    conversation_id = request.conversation_id

    # Create conversation if needed and teen_id is provided
    if not conversation_id and request.teen_id:
        title = generate_title_from_message(last_user_message)
        conversation_id = await create_conversation(request.teen_id, title)
        if conversation_id:
            logger.info(f"📝 Created new conversation: {conversation_id}")

    # Save user message (if we have a conversation)
    user_msg_saved = False
    if conversation_id:
        user_msg_saved = await save_message(
            conversation_id=conversation_id,
            role="user",
            content=last_user_message,
            tier=tier,
            categories=topics,
        )
        if user_msg_saved:
            logger.info(f"💾 User message saved to conversation: {conversation_id}")
        else:
            logger.warning(f"⚠️ Failed to save user message to conversation: {conversation_id}")

    # ==========================================
    # STEP 4: Generate AI Response (Tier 1 only)
    # ==========================================
    logger.info(f"✅ Tier 1 (GREEN): Routing to AI provider")

    # Add system prompt for age-appropriate responses
    age = request.user_age or 15
    system_prompt = get_system_prompt_for_age(age)

    messages_with_system = [ChatMessage(role="system", content=system_prompt)]
    messages_with_system.extend(request.messages)

    try:
        ai_response = await route_to_ai(
            messages=messages_with_system,
            model=request.model,
            temperature=request.temperature or 0.7,
            max_tokens=request.max_tokens or 1000,
        )
    except httpx.HTTPStatusError as e:
        logger.error(f"AI provider error: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"AI service temporarily unavailable. Please try again or select a different model."
        )
    except Exception as e:
        logger.error(f"AI routing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # ==========================================
    # STEP 5: Post-Classification (AI Response Check)
    # ==========================================
    response_tier = 1
    try:
        response_classification = await classifier.classify(
            message=ai_response,
            teen_id=request.teen_id,
        )
        response_tier = response_classification.tier

        # If AI response is problematic, filter it
        if response_classification.tier >= 3:
            logger.warning(f"⚠️ AI response classified as Tier {response_classification.tier}, filtering")
            ai_response = "I apologize, but I need to be more careful with my response. Let me try a different approach. What specific aspect would you like help with?"
    except Exception as e:
        logger.error(f"Post-classification error: {e}")
        # Continue with original response

    # ==========================================
    # STEP 6: Save AI Response to Message Service
    # ==========================================
    assistant_msg_saved = False
    if conversation_id:
        assistant_msg_saved = await save_message(
            conversation_id=conversation_id,
            role="assistant",
            content=ai_response,
            tier=response_tier,
            categories=topics,
        )
        if assistant_msg_saved:
            logger.info(f"💾 AI response saved to conversation: {conversation_id}")
        else:
            logger.warning(f"⚠️ Failed to save AI response to conversation: {conversation_id}")

    # ==========================================
    # STEP 7: Return Response
    # ==========================================
    processing_time = (time.time() - start_time) * 1000
    logger.info(f"✅ Response generated in {processing_time:.0f}ms")

    return ChatCompletionResponse(
        id=f"chatcmpl-{int(time.time())}",
        created=int(time.time()),
        model=request.model,
        choices=[ChatCompletionChoice(
            message=ChatMessage(role="assistant", content=ai_response),
            finish_reason="stop",
        )],
        usage=ChatCompletionUsage(
            prompt_tokens=len(last_user_message.split()),
            completion_tokens=len(ai_response.split()),
            total_tokens=len(last_user_message.split()) + len(ai_response.split()),
        ),
        safety=safety_metadata,
        # Include conversation_id so client can continue the conversation
        conversation_id=conversation_id,
    )


@chat_router.get("/models", response_model=ModelListResponse)
async def list_models():
    """List available models (OpenAI-compatible)"""
    # Note: OpenAI models disabled - only using Gemini and Claude
    return ModelListResponse(
        data=[
            ModelInfo(id="modai-safe", owned_by="modai"),  # Routes to Gemini
            ModelInfo(id="gemini-2.0-flash-exp", owned_by="google"),
            ModelInfo(id="claude-sonnet-4-5-20250929", owned_by="anthropic"),
            ModelInfo(id="claude-3-5-haiku-20241022", owned_by="anthropic"),
        ]
    )


@chat_router.get("/models/{model_id}")
async def get_model(model_id: str):
    """Get model info (OpenAI-compatible)"""
    if model_id not in MODEL_PROVIDERS:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

    return ModelInfo(id=model_id, owned_by=MODEL_PROVIDERS.get(model_id, "modai"))
