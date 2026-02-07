"""
Response Generator for Tier-Based Guidance

Generates contextual, empathetic responses for different safety tiers.
Tier 2 (YELLOW) provides gentle redirection while allowing conversation to continue.
"""

from typing import Optional


class ResponseGenerator:
    """Generates tier-appropriate responses for topic classification."""
    
    # Tier 2 (YELLOW) - Guidance without blocking
    TIER_2_TEMPLATES = {
        "mental_health": (
            "I notice you want to discuss {topic}. That's a really important topic! "
            "While I can offer some general support, I want to make sure you get the best help. "
            "Have you considered talking to a school counselor or trusted adult about this? "
            "In the meantime, I'm here to discuss study strategies, stress management, or anything else on your mind. "
            "What would you like to focus on?"
        ),
        "relationships": (
            "Relationships can definitely be complex! I'm here to help you think through things. "
            "Just a reminder - if you ever feel unsafe or uncomfortable in any situation, "
            "please talk to a trusted adult. For general relationship questions, I'm happy to discuss! "
            "What specifically would you like guidance on?"
        ),
        "health": (
            "Health topics are important to discuss! For specific medical concerns, "
            "I'd recommend talking to a parent, guardian, or school nurse. "
            "I can definitely help with general wellness, exercise tips, or healthy habits though! "
            "What aspect of health are you curious about?"
        ),
        "default": (
            "I can help with {topic}! Just want to make sure we approach this thoughtfully. "
            "If this becomes something more serious, remember there are trusted adults and resources available. "
            "For now, let's explore this together - what would you like to know?"
        ),
    }
    
    # Tier 3 (ORANGE) - Alert supervisor, continue conversation
    TIER_3_MESSAGE = (
        "I'm here to support you with {topic}. "
        "I've let your parent/guardian know you're discussing this topic so they can offer additional support if needed. "
        "You're not in trouble - we just want to make sure you have all the help you need. "
        "How can I help you with this today?"
    )
    
    # Tier 4 (RED) - Crisis, block and provide resources + Legal compliance
    TIER_4_MESSAGE = (
        "**I'm concerned about your safety and well-being.**\\n\\n"
        "While I'm here to support you, what you've shared suggests you may need immediate help from a qualified professional. "
        "**Please reach out to a trusted adult, school counselor, or contact one of these confidential resources:**\\n\\n"
        "📞 **988 Suicide & Crisis Lifeline** - Available 24/7, call or text 988\\n"
        "💬 **Crisis Text Line** - Text HOME to 741741\\n"
        "🌐 **Trevor Project** (LGBTQ+ youth) - 1-866-488-7386\\n\\n"
        "**Important:** I've also notified your parent/guardian so they can provide additional support. "
        "This is not about getting you in trouble - it's about making sure you're safe.\\n\\n"
        "If you're in immediate danger, please call 911 or go to your nearest emergency room.\\n\\n"
        "Your safety is the top priority. I'm here when you're ready to talk about something else."
    )
    
    @staticmethod
    def get_tier_2_guidance(topic_name: str, topic_category: Optional[str] = None) -> str:
        """
        Generate gentle guidance for Tier 2 (YELLOW) topics.
        
        Args:
            topic_name: Human-readable topic name
            topic_category: Category for template selection
            
        Returns:
            Contextual guidance message
        """
        # Normalize category for template lookup
        category_key = "default"
        if topic_category:
            cat_lower = topic_category.lower()
            if "mental" in cat_lower or "health" in cat_lower:
                category_key = "mental_health"
            elif "relationship" in cat_lower or "dating" in cat_lower:
                category_key = "relationships"
            elif "physical" in cat_lower or "body" in cat_lower:
                category_key = "health"
        
        template = ResponseGenerator.TIER_2_TEMPLATES.get(category_key, ResponseGenerator.TIER_2_TEMPLATES["default"])
        return template.format(topic=topic_name.lower())
    
    @staticmethod
    def get_tier_3_alert(topic_name: str) -> str:
        """Generate supervisor alert message for Tier 3 (ORANGE) topics."""
        return ResponseGenerator.TIER_3_MESSAGE.format(topic=topic_name.lower())
    
    @staticmethod
    def get_tier_4_crisis() -> str:
        """Generate crisis resources message for Tier 4 (RED) topics."""
        return ResponseGenerator.TIER_4_MESSAGE
    
    @staticmethod
    def get_disabled_topic_message(topic_name: str) -> str:
        """Generate empathetic message for supervisor-disabled topics."""
        return (
            f"This topic ({topic_name}) has been disabled by your supervisor.\n\n"
            "If you believe this should be allowed, please talk to your parent/guardian. "
            "They can adjust topic permissions in their dashboard.\n\n"
            "Is there something else I can help you with? 😊"
        )
    
    @staticmethod
    def get_tier_override_message(original_tier: int, new_tier: int, topic_name: str) -> str:
        """Generate message when supervisor overrides tier to more restrictive level."""
        if new_tier == 4:
            return ResponseGenerator.get_tier_4_crisis()
        elif new_tier == 3:
            return ResponseGenerator.get_tier_3_alert(topic_name)
        elif new_tier == 2:
            return ResponseGenerator.get_tier_2_guidance(topic_name)
        else:
            return f"Topic restrictions have been updated for {topic_name}."
