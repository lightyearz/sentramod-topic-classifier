"""
ModAI Hierarchical Topic Taxonomy

A comprehensive multi-level topic hierarchy for the 4-tier safety system.
Topics can share labels and belong to multiple categories using set-based classification.

Usage:
    from models.TOPIC_TAXONOMY import TAXONOMY, classify_message_by_topics

    result = classify_message_by_topics("Can you help me with algebra homework?")
    # Returns: {
    #   "tier": 1,
    #   "topics": ["homework", "math", "algebra"],
    #   "hierarchy": "Academic > Homework Help > Math > Algebra",
    #   "labels_matched": ["algebra", "homework", "help"],
    #   "confidence": 0.92
    # }
"""

from dataclasses import dataclass, field
from typing import List, Set, Optional, Dict, Any
from enum import Enum


class Tier(Enum):
    """Safety tier classification."""

    GREEN = 1  # Always allowed
    YELLOW = 2  # Needs approval
    ORANGE = 3  # Requires supervision
    RED = 4  # Auto-blocked / Crisis


@dataclass
class Topic:
    """
    Hierarchical topic with shared labels.

    Topics can:
    - Have multiple parents (e.g., "stress" under both "school" and "mental health")
    - Share labels with other topics (e.g., "anxious" in both anxiety and stress)
    - Inherit tier from parent or override
    """

    id: str
    name: str
    tier: Tier
    level: int  # 1=root, 2=category, 3=subcategory, 4=specific
    parent_ids: List[str] = field(default_factory=list)  # Can have multiple parents
    labels: Set[str] = field(
        default_factory=set
    )  # Use set for fast lookup and deduplication
    description: str = ""
    emoji: str = ""
    dashboard_color: str = ""  # CSS color for dashboard display
    requires_model: Optional[str] = None  # Which specialized model to use


# =============================================================================
# TIER 1 - ACADEMIC & SAFE (GREEN) 🟢
# =============================================================================

TIER_1_TOPICS = [
    # Root: Academic
    Topic(
        id="academic",
        name="Academic",
        tier=Tier.GREEN,
        level=1,
        parent_ids=[],
        labels={"academic", "school", "education", "learning", "study"},
        description="School-related academic topics",
        emoji="📚",
        dashboard_color="#10B981",  # Green
    ),
    # -------------------------------------------------------------------------
    # Homework Help
    # -------------------------------------------------------------------------
    Topic(
        id="homework",
        name="Homework Help",
        tier=Tier.GREEN,
        level=2,
        parent_ids=["academic"],
        labels={"homework", "assignment", "help", "can you help", "help me"},
        description="General homework assistance",
        emoji="📝",
        dashboard_color="#10B981",
    ),
    # Math
    Topic(
        id="math",
        name="Mathematics",
        tier=Tier.GREEN,
        level=3,
        parent_ids=["homework", "academic"],
        labels={"math", "mathematics", "calculate", "solve", "equation"},
        description="Math-related questions",
        emoji="🔢",
        dashboard_color="#10B981",
    ),
    Topic(
        id="algebra",
        name="Algebra",
        tier=Tier.GREEN,
        level=4,
        parent_ids=["math"],
        labels={
            "algebra",
            "solve for x",
            "variable",
            "polynomial",
            "quadratic",
            "linear equation",
        },
        description="Algebra problems and equations",
        emoji="✖️",
        dashboard_color="#10B981",
    ),
    Topic(
        id="geometry",
        name="Geometry",
        tier=Tier.GREEN,
        level=4,
        parent_ids=["math"],
        labels={
            "geometry",
            "triangle",
            "circle",
            "angle",
            "shape",
            "area",
            "volume",
            "perimeter",
        },
        description="Geometry problems",
        emoji="📐",
        dashboard_color="#10B981",
    ),
    Topic(
        id="calculus",
        name="Calculus",
        tier=Tier.GREEN,
        level=4,
        parent_ids=["math"],
        labels={
            "calculus",
            "derivative",
            "integral",
            "limit",
            "differentiation",
            "integration",
        },
        description="Calculus problems",
        emoji="∫",
        dashboard_color="#10B981",
    ),
    # Science
    Topic(
        id="science",
        name="Science",
        tier=Tier.GREEN,
        level=3,
        parent_ids=["homework", "academic"],
        labels={"science", "experiment", "hypothesis", "scientific"},
        description="Science questions",
        emoji="🔬",
        dashboard_color="#10B981",
    ),
    Topic(
        id="biology",
        name="Biology",
        tier=Tier.GREEN,
        level=4,
        parent_ids=["science"],
        labels={
            "biology",
            "cell",
            "DNA",
            "organism",
            "photosynthesis",
            "evolution",
            "ecosystem",
        },
        description="Biology topics",
        emoji="🧬",
        dashboard_color="#10B981",
    ),
    Topic(
        id="chemistry",
        name="Chemistry",
        tier=Tier.GREEN,
        level=4,
        parent_ids=["science"],
        labels={
            "chemistry",
            "chemical",
            "element",
            "compound",
            "reaction",
            "molecule",
            "atom",
        },
        description="Chemistry topics",
        emoji="⚗️",
        dashboard_color="#10B981",
    ),
    Topic(
        id="physics",
        name="Physics",
        tier=Tier.GREEN,
        level=4,
        parent_ids=["science"],
        labels={
            "physics",
            "force",
            "energy",
            "motion",
            "velocity",
            "gravity",
            "momentum",
        },
        description="Physics topics",
        emoji="⚛️",
        dashboard_color="#10B981",
    ),
    # English / Language Arts
    Topic(
        id="english",
        name="English / Language Arts",
        tier=Tier.GREEN,
        level=3,
        parent_ids=["homework", "academic"],
        labels={"english", "writing", "grammar", "essay", "reading", "literature"},
        description="English and writing help",
        emoji="📖",
        dashboard_color="#10B981",
    ),
    Topic(
        id="writing",
        name="Writing Help",
        tier=Tier.GREEN,
        level=4,
        parent_ids=["english"],
        labels={
            "write",
            "essay",
            "paragraph",
            "thesis",
            "outline",
            "draft",
            "proofread",
        },
        description="Writing assistance",
        emoji="✍️",
        dashboard_color="#10B981",
    ),
    Topic(
        id="literature",
        name="Literature Analysis",
        tier=Tier.GREEN,
        level=4,
        parent_ids=["english"],
        labels={"book", "novel", "poem", "author", "character", "theme", "analysis"},
        description="Literature analysis and discussion",
        emoji="📚",
        dashboard_color="#10B981",
    ),
    # History
    Topic(
        id="history",
        name="History",
        tier=Tier.GREEN,
        level=3,
        parent_ids=["homework", "academic"],
        labels={"history", "historical", "war", "president", "ancient", "civilization"},
        description="History topics",
        emoji="🏛️",
        dashboard_color="#10B981",
    ),
    # -------------------------------------------------------------------------
    # Career & Future
    # -------------------------------------------------------------------------
    Topic(
        id="career",
        name="Career Planning",
        tier=Tier.GREEN,
        level=2,
        parent_ids=["academic"],
        labels={
            "career",
            "job",
            "profession",
            "future",
            "college",
            "university",
            "major",
        },
        description="Career exploration and planning",
        emoji="💼",
        dashboard_color="#10B981",
    ),
    Topic(
        id="programming",
        name="Programming / Tech Careers",
        tier=Tier.GREEN,
        level=3,
        parent_ids=["career"],
        labels={
            "programming",
            "coding",
            "developer",
            "software",
            "engineer",
            "computer science",
        },
        description="Technology and programming careers",
        emoji="💻",
        dashboard_color="#10B981",
    ),
    # -------------------------------------------------------------------------
    # Hobbies & Interests
    # -------------------------------------------------------------------------
    Topic(
        id="hobbies",
        name="Hobbies & Interests",
        tier=Tier.GREEN,
        level=2,
        parent_ids=[],
        labels={"hobby", "hobbies", "interest", "activity", "fun"},
        description="Personal interests and activities",
        emoji="🎨",
        dashboard_color="#10B981",
    ),
    Topic(
        id="sports",
        name="Sports",
        tier=Tier.GREEN,
        level=3,
        parent_ids=["hobbies"],
        labels={"sport", "sports", "team", "game", "practice", "athlete", "exercise"},
        description="Sports and athletics",
        emoji="⚽",
        dashboard_color="#10B981",
    ),
    Topic(
        id="arts",
        name="Arts & Music",
        tier=Tier.GREEN,
        level=3,
        parent_ids=["hobbies"],
        labels={"art", "draw", "paint", "music", "sing", "play instrument", "creative"},
        description="Creative arts and music",
        emoji="🎵",
        dashboard_color="#10B981",
    ),
    Topic(
        id="gaming",
        name="Gaming",
        tier=Tier.GREEN,
        level=3,
        parent_ids=["hobbies"],
        labels={"game", "gaming", "video game", "play", "gamer"},
        description="Video games and gaming",
        emoji="🎮",
        dashboard_color="#10B981",
    ),
]


# =============================================================================
# TIER 2 - SENSITIVE TOPICS (YELLOW) 🟡
# =============================================================================

TIER_2_TOPICS = [
    # Root: Social Development
    Topic(
        id="social",
        name="Social Development",
        tier=Tier.YELLOW,
        level=1,
        parent_ids=[],
        labels={"social", "friends", "peer", "relationships"},
        description="Social relationships and development",
        emoji="👥",
        dashboard_color="#F59E0B",  # Yellow
        requires_model="unitary/toxic-bert",
    ),
    # -------------------------------------------------------------------------
    # Dating & Relationships
    # -------------------------------------------------------------------------
    Topic(
        id="dating",
        name="Dating & Relationships",
        tier=Tier.YELLOW,
        level=2,
        parent_ids=["social"],
        labels={
            "dating",
            "date",
            "boyfriend",
            "girlfriend",
            "crush",
            "like someone",
            "relationship",
        },
        description="Teen dating and romantic relationships",
        emoji="💕",
        dashboard_color="#F59E0B",
        requires_model="bart-zero-shot",
    ),
    Topic(
        id="first_crush",
        name="First Crush",
        tier=Tier.YELLOW,
        level=3,
        parent_ids=["dating"],
        labels={"crush", "like them", "have feelings", "attracted to"},
        description="First crush experiences",
        emoji="😊",
        dashboard_color="#F59E0B",
    ),
    Topic(
        id="asking_out",
        name="Asking Someone Out",
        tier=Tier.YELLOW,
        level=3,
        parent_ids=["dating"],
        labels={"ask out", "ask them", "how do i talk to", "what should i say"},
        description="How to express interest",
        emoji="💬",
        dashboard_color="#F59E0B",
    ),
    Topic(
        id="breakup",
        name="Breakups",
        tier=Tier.YELLOW,
        level=3,
        parent_ids=["dating"],
        labels={"breakup", "broke up", "ex", "break up with", "ended relationship"},
        description="Relationship endings",
        emoji="💔",
        dashboard_color="#F59E0B",
    ),
    # -------------------------------------------------------------------------
    # Body Image & Appearance
    # -------------------------------------------------------------------------
    Topic(
        id="body_image",
        name="Body Image",
        tier=Tier.YELLOW,
        level=2,
        parent_ids=["social"],
        labels={
            "body",
            "appearance",
            "look",
            "weight",
            "skinny",
            "fat",
            "ugly",
            "pretty",
        },
        description="Body image and appearance concerns",
        emoji="🪞",
        dashboard_color="#F59E0B",
        requires_model="bart-zero-shot",
    ),
    Topic(
        id="weight_concerns",
        name="Weight Concerns",
        tier=Tier.YELLOW,
        level=3,
        parent_ids=["body_image"],
        labels={"too skinny", "too fat", "lose weight", "gain weight", "body weight"},
        description="Weight and body size concerns",
        emoji="⚖️",
        dashboard_color="#F59E0B",
    ),
    Topic(
        id="appearance",
        name="Appearance Worries",
        tier=Tier.YELLOW,
        level=3,
        parent_ids=["body_image"],
        labels={"ugly", "not pretty", "not handsome", "look bad", "appearance"},
        description="Concerns about physical appearance",
        emoji="😔",
        dashboard_color="#F59E0B",
    ),
    # -------------------------------------------------------------------------
    # Social Media
    # -------------------------------------------------------------------------
    Topic(
        id="social_media",
        name="Social Media",
        tier=Tier.YELLOW,
        level=2,
        parent_ids=["social"],
        labels={
            "social media",
            "instagram",
            "tiktok",
            "snapchat",
            "followers",
            "likes",
            "post",
        },
        description="Social media usage and concerns",
        emoji="📱",
        dashboard_color="#F59E0B",
        requires_model="bart-zero-shot",
    ),
    Topic(
        id="social_media_pressure",
        name="Social Media Pressure",
        tier=Tier.YELLOW,
        level=3,
        parent_ids=["social_media"],
        labels={
            "everyone has",
            "my friends are on",
            "parents won't let me",
            "can't have",
        },
        description="Pressure around social media use",
        emoji="😰",
        dashboard_color="#F59E0B",
    ),
    # -------------------------------------------------------------------------
    # Peer Pressure
    # -------------------------------------------------------------------------
    Topic(
        id="peer_pressure",
        name="Peer Pressure",
        tier=Tier.YELLOW,
        level=2,
        parent_ids=["social"],
        labels={
            "peer pressure",
            "friends want me",
            "everyone is doing",
            "want me to",
            "pressure",
        },
        description="Pressure from peers",
        emoji="🤝",
        dashboard_color="#F59E0B",
        requires_model="bart-zero-shot",
    ),
    # -------------------------------------------------------------------------
    # Friendship Issues
    # -------------------------------------------------------------------------
    Topic(
        id="friendship",
        name="Friendship Issues",
        tier=Tier.YELLOW,
        level=2,
        parent_ids=["social"],
        labels={
            "friend",
            "friendship",
            "best friend",
            "fight with friend",
            "friend drama",
        },
        description="Friendship conflicts and issues",
        emoji="👭",
        dashboard_color="#F59E0B",
        requires_model="bart-zero-shot",
    ),
]


# =============================================================================
# TIER 3 - MENTAL HEALTH (ORANGE) 🟠
# =============================================================================

TIER_3_TOPICS = [
    # Root: Mental Health
    Topic(
        id="mental_health",
        name="Mental Health",
        tier=Tier.ORANGE,
        level=1,
        parent_ids=[],
        labels={"mental health", "feeling", "emotion", "mental", "psychological"},
        description="Mental health and emotional wellbeing",
        emoji="🧠",
        dashboard_color="#F97316",  # Orange
        requires_model="j-hartmann/emotion-english-distilroberta-base",
    ),
    # -------------------------------------------------------------------------
    # Anxiety & Stress
    # -------------------------------------------------------------------------
    Topic(
        id="anxiety",
        name="Anxiety",
        tier=Tier.ORANGE,
        level=2,
        parent_ids=["mental_health"],
        labels={
            "anxiety",
            "anxious",
            "worried",
            "nervous",
            "panic",
            "stress",
            "stressed",
        },
        description="Anxiety and stress concerns",
        emoji="😰",
        dashboard_color="#F97316",
        requires_model="j-hartmann/emotion-english-distilroberta-base",
    ),
    Topic(
        id="school_anxiety",
        name="School Anxiety",
        tier=Tier.ORANGE,
        level=3,
        parent_ids=["anxiety", "academic"],  # Multiple parents!
        labels={
            "anxious about school",
            "stressed about test",
            "worried about grades",
            "school stress",
        },
        description="Anxiety related to school",
        emoji="📚😰",
        dashboard_color="#F97316",
    ),
    Topic(
        id="social_anxiety",
        name="Social Anxiety",
        tier=Tier.ORANGE,
        level=3,
        parent_ids=["anxiety", "social"],  # Multiple parents!
        labels={
            "social anxiety",
            "scared to talk",
            "afraid of people",
            "nervous around others",
        },
        description="Anxiety in social situations",
        emoji="👥😰",
        dashboard_color="#F97316",
    ),
    Topic(
        id="panic_attacks",
        name="Panic Attacks",
        tier=Tier.ORANGE,
        level=3,
        parent_ids=["anxiety"],
        labels={
            "panic attack",
            "can't breathe",
            "heart racing",
            "chest tight",
            "panic",
        },
        description="Panic attack symptoms",
        emoji="😱",
        dashboard_color="#F97316",
    ),
    # -------------------------------------------------------------------------
    # Depression & Sadness
    # -------------------------------------------------------------------------
    Topic(
        id="depression",
        name="Depression",
        tier=Tier.ORANGE,
        level=2,
        parent_ids=["mental_health"],
        labels={
            "depression",
            "depressed",
            "sad",
            "sadness",
            "hopeless",
            "empty",
            "numb",
        },
        description="Depression and persistent sadness",
        emoji="😔",
        dashboard_color="#F97316",
        requires_model="malexandersalazar/xlm-roberta-base-cls-depression",
    ),
    Topic(
        id="loneliness",
        name="Loneliness",
        tier=Tier.ORANGE,
        level=3,
        parent_ids=["depression"],
        labels={
            "lonely",
            "alone",
            "no one understands",
            "nobody cares",
            "isolated",
            "feel alone",
        },
        description="Feelings of loneliness and isolation",
        emoji="😞",
        dashboard_color="#F97316",
    ),
    Topic(
        id="lack_motivation",
        name="Lack of Motivation",
        tier=Tier.ORANGE,
        level=3,
        parent_ids=["depression"],
        labels={
            "no motivation",
            "can't do anything",
            "no energy",
            "tired all the time",
            "exhausted",
        },
        description="Loss of motivation and energy",
        emoji="😴",
        dashboard_color="#F97316",
    ),
    # -------------------------------------------------------------------------
    # Bullying & Harassment
    # -------------------------------------------------------------------------
    Topic(
        id="bullying",
        name="Bullying",
        tier=Tier.ORANGE,
        level=2,
        parent_ids=["mental_health", "social"],  # Multiple parents!
        labels={
            "bully",
            "bullying",
            "bullied",
            "make fun of",
            "pick on",
            "tease",
            "harass",
        },
        description="Bullying and harassment",
        emoji="😢",
        dashboard_color="#F97316",
        requires_model="unitary/toxic-bert",
    ),
    Topic(
        id="cyberbullying",
        name="Cyberbullying",
        tier=Tier.ORANGE,
        level=3,
        parent_ids=["bullying", "social_media"],  # Multiple parents!
        labels={
            "cyberbully",
            "online bullying",
            "mean comments",
            "hate messages",
            "online harassment",
        },
        description="Online bullying and harassment",
        emoji="📱😢",
        dashboard_color="#F97316",
    ),
    Topic(
        id="name_calling",
        name="Name Calling & Insults",
        tier=Tier.ORANGE,
        level=3,
        parent_ids=["bullying"],
        labels={"call me names", "insult", "making fun of me", "laugh at me", "mock"},
        description="Verbal bullying and insults",
        emoji="😠",
        dashboard_color="#F97316",
    ),
    # -------------------------------------------------------------------------
    # Family Conflict
    # -------------------------------------------------------------------------
    Topic(
        id="family",
        name="Family Issues",
        tier=Tier.ORANGE,
        level=2,
        parent_ids=["mental_health"],
        labels={"family", "parents", "mom", "dad", "sibling", "brother", "sister"},
        description="Family relationships and conflicts",
        emoji="👨‍👩‍👧‍👦",
        dashboard_color="#F97316",
        requires_model="j-hartmann/emotion-english-distilroberta-base",
    ),
    Topic(
        id="parent_conflict",
        name="Parent Conflict",
        tier=Tier.ORANGE,
        level=3,
        parent_ids=["family"],
        labels={
            "parents fighting",
            "parents argue",
            "parents yelling",
            "divorce",
            "parents separated",
        },
        description="Conflict between parents",
        emoji="💔",
        dashboard_color="#F97316",
    ),
    Topic(
        id="parent_child_conflict",
        name="Conflict with Parents",
        tier=Tier.ORANGE,
        level=3,
        parent_ids=["family"],
        labels={
            "fight with parents",
            "parents don't understand",
            "parents mad at me",
            "grounded",
        },
        description="Conflict with parents",
        emoji="😤",
        dashboard_color="#F97316",
    ),
    # -------------------------------------------------------------------------
    # Sleep Issues
    # -------------------------------------------------------------------------
    Topic(
        id="sleep",
        name="Sleep Problems",
        tier=Tier.ORANGE,
        level=2,
        parent_ids=["mental_health"],
        labels={
            "can't sleep",
            "insomnia",
            "trouble sleeping",
            "sleep problems",
            "tired",
            "exhausted",
        },
        description="Sleep difficulties",
        emoji="😴",
        dashboard_color="#F97316",
    ),
]


# =============================================================================
# TIER 4 - CRISIS (RED) 🔴
# =============================================================================

TIER_4_TOPICS = [
    # Root: Crisis Situations
    Topic(
        id="crisis",
        name="Crisis Situations",
        tier=Tier.RED,
        level=1,
        parent_ids=[],
        labels={"crisis", "emergency", "danger", "help me"},
        description="Crisis situations requiring immediate intervention",
        emoji="🆘",
        dashboard_color="#EF4444",  # Red
        requires_model="sentinet/suicidality",
    ),
    # -------------------------------------------------------------------------
    # Suicide Ideation
    # -------------------------------------------------------------------------
    Topic(
        id="suicide",
        name="Suicide Ideation",
        tier=Tier.RED,
        level=2,
        parent_ids=["crisis"],
        labels={
            "suicide",
            "suicidal",
            "kill myself",
            "end my life",
            "don't want to live",
            "want to die",
            "better off dead",
            "end it all",
            "not worth living",
            "don't want to be here",
            "wish i was dead",
        },
        description="Suicide thoughts and ideation - IMMEDIATE INTERVENTION",
        emoji="🆘",
        dashboard_color="#EF4444",
        requires_model="sentinet/suicidality",
    ),
    Topic(
        id="suicide_methods",
        name="Suicide Methods",
        tier=Tier.RED,
        level=3,
        parent_ids=["suicide"],
        labels={
            "how to kill myself",
            "ways to die",
            "suicide method",
            "painless death",
        },
        description="Searching for suicide methods - CRITICAL",
        emoji="⚠️",
        dashboard_color="#EF4444",
        requires_model="llm-guard-bantopics",
    ),
    # -------------------------------------------------------------------------
    # Self-Harm
    # -------------------------------------------------------------------------
    Topic(
        id="self_harm",
        name="Self-Harm",
        tier=Tier.RED,
        level=2,
        parent_ids=["crisis"],
        labels={
            "self harm",
            "self-harm",
            "cut myself",
            "cutting",
            "hurt myself",
            "burn myself",
            "harm myself",
            "self injury",
            "want to hurt myself",
        },
        description="Self-harm behaviors - IMMEDIATE INTERVENTION",
        emoji="🆘",
        dashboard_color="#EF4444",
        requires_model="sentinet/suicidality",
    ),
    Topic(
        id="cutting",
        name="Cutting",
        tier=Tier.RED,
        level=3,
        parent_ids=["self_harm"],
        labels={"cut", "cutting", "blade", "razor", "wrist", "arm"},
        description="Cutting behavior",
        emoji="⚠️",
        dashboard_color="#EF4444",
    ),
    # -------------------------------------------------------------------------
    # Violence & Threats
    # -------------------------------------------------------------------------
    Topic(
        id="violence",
        name="Violence",
        tier=Tier.RED,
        level=2,
        parent_ids=["crisis"],
        labels={
            "violence",
            "violent",
            "hurt someone",
            "kill them",
            "shoot",
            "stab",
            "weapon",
            "gun",
            "knife",
            "attack",
            "fight",
        },
        description="Violence and threats toward others",
        emoji="⚠️",
        dashboard_color="#EF4444",
        requires_model="unitary/toxic-bert",
    ),
    Topic(
        id="school_threat",
        name="School Violence Threats",
        tier=Tier.RED,
        level=3,
        parent_ids=["violence"],
        labels={
            "school shooting",
            "hurt people at school",
            "attack school",
            "weapon to school",
        },
        description="Threats toward school - REPORT IMMEDIATELY",
        emoji="🚨",
        dashboard_color="#EF4444",
        requires_model="llm-guard-bantopics",
    ),
    # -------------------------------------------------------------------------
    # Substance Abuse
    # -------------------------------------------------------------------------
    Topic(
        id="substance",
        name="Substance Abuse",
        tier=Tier.RED,
        level=2,
        parent_ids=["crisis"],
        labels={
            "drugs",
            "drug",
            "alcohol",
            "drinking",
            "drunk",
            "high",
            "weed",
            "marijuana",
            "cocaine",
            "pills",
            "overdose",
            "substance",
        },
        description="Substance abuse and drug use",
        emoji="💊",
        dashboard_color="#EF4444",
        requires_model="llm-guard-bantopics",
    ),
    Topic(
        id="peer_substance",
        name="Peer Substance Pressure",
        tier=Tier.RED,
        level=3,
        parent_ids=["substance", "peer_pressure"],  # Multiple parents!
        labels={
            "friends drinking",
            "try drugs",
            "party drinking",
            "want me to drink",
            "peer pressure drugs",
        },
        description="Pressure to use substances",
        emoji="🍺",
        dashboard_color="#EF4444",
    ),
    # -------------------------------------------------------------------------
    # Sexual Content
    # -------------------------------------------------------------------------
    Topic(
        id="sexual_content",
        name="Sexual Content",
        tier=Tier.RED,
        level=2,
        parent_ids=["crisis"],
        labels={
            "sex",
            "sexual",
            "virginity",
            "hook up",
            "nude",
            "naked",
            "sexual abuse",
            "molest",
            "rape",
            "assault",
        },
        description="Sexual content and abuse - SENSITIVE",
        emoji="⚠️",
        dashboard_color="#EF4444",
        requires_model="llm-guard-bantopics",
    ),
    Topic(
        id="sexual_abuse",
        name="Sexual Abuse",
        tier=Tier.RED,
        level=3,
        parent_ids=["sexual_content"],
        labels={
            "sexual abuse",
            "molest",
            "rape",
            "assault",
            "touched me",
            "inappropriate touch",
        },
        description="Sexual abuse - REPORT TO AUTHORITIES",
        emoji="🚨",
        dashboard_color="#EF4444",
    ),
    # -------------------------------------------------------------------------
    # Eating Disorders
    # -------------------------------------------------------------------------
    Topic(
        id="eating_disorder",
        name="Eating Disorders",
        tier=Tier.RED,
        level=2,
        parent_ids=["crisis", "body_image"],  # Multiple parents!
        labels={
            "anorexia",
            "bulimia",
            "eating disorder",
            "starve",
            "not eating",
            "throw up food",
            "purge",
            "binge",
        },
        description="Eating disorder behaviors - MEDICAL EMERGENCY",
        emoji="🆘",
        dashboard_color="#EF4444",
        requires_model="j-hartmann/emotion-english-distilroberta-base",
    ),
]


# =============================================================================
# Complete Taxonomy
# =============================================================================

ALL_TOPICS = TIER_1_TOPICS + TIER_2_TOPICS + TIER_3_TOPICS + TIER_4_TOPICS

# Create lookup dictionaries for fast access
TOPICS_BY_ID = {topic.id: topic for topic in ALL_TOPICS}
TOPICS_BY_TIER = {
    Tier.GREEN: TIER_1_TOPICS,
    Tier.YELLOW: TIER_2_TOPICS,
    Tier.ORANGE: TIER_3_TOPICS,
    Tier.RED: TIER_4_TOPICS,
}


# =============================================================================
# Label Index (for fast set-based classification)
# =============================================================================


def build_label_index() -> Dict[str, Set[str]]:
    """
    Build reverse index: label -> set of topic IDs.

    This allows fast lookup: which topics match these labels?
    Uses sets for O(1) membership testing and union operations.
    """
    label_index = {}

    for topic in ALL_TOPICS:
        for label in topic.labels:
            if label not in label_index:
                label_index[label] = set()
            label_index[label].add(topic.id)

    return label_index


LABEL_INDEX = build_label_index()


# =============================================================================
# Convenience Exports (for backward compatibility with classifier service)
# =============================================================================

# Export TOPICS as alias for ALL_TOPICS
TOPICS = ALL_TOPICS

# Export all unique labels
ALL_LABELS = set()
for topic in ALL_TOPICS:
    ALL_LABELS.update(topic.labels)


def get_topic_by_id(topic_id: str) -> Optional[Topic]:
    """Get a topic by its ID."""
    return TOPICS_BY_ID.get(topic_id)


def get_tier_topics(tier: int) -> List[Topic]:
    """
    Get all topics for a specific tier.

    Args:
        tier: Tier number (1=GREEN, 2=YELLOW, 3=ORANGE, 4=RED)

    Returns:
        List of topics in that tier
    """
    tier_enum = Tier(tier)
    return TOPICS_BY_TIER.get(tier_enum, [])


# =============================================================================
# Classification Functions
# =============================================================================


def get_topic_hierarchy(topic_id: str) -> List[str]:
    """
    Get full hierarchy path for a topic.

    Example: get_topic_hierarchy("algebra")
    Returns: ["academic", "homework", "math", "algebra"]
    """
    if topic_id not in TOPICS_BY_ID:
        return []

    topic = TOPICS_BY_ID[topic_id]
    hierarchy = [topic_id]

    # Build hierarchy by following parent links
    # If multiple parents, take the first one for primary path
    current = topic
    while current.parent_ids:
        parent_id = current.parent_ids[0]  # Primary parent
        hierarchy.insert(0, parent_id)
        current = TOPICS_BY_ID.get(parent_id)
        if not current:
            break

    return hierarchy


def get_topic_hierarchy_display(topic_id: str) -> str:
    """
    Get hierarchy as display string.

    Example: "Academic > Homework Help > Math > Algebra"
    """
    hierarchy = get_topic_hierarchy(topic_id)
    names = [TOPICS_BY_ID[tid].name for tid in hierarchy if tid in TOPICS_BY_ID]
    return " > ".join(names)


def classify_by_labels(detected_labels: List[str]) -> Dict[str, Any]:
    """
    Classify message based on detected labels using set operations.

    Args:
        detected_labels: List of labels detected in message

    Returns:
        Classification result with tier, matched topics, and hierarchy
    """
    # Convert to set for fast operations
    detected_set = set(label.lower() for label in detected_labels)

    # Find all topics that match any detected labels
    matched_topic_ids = set()
    matched_labels = set()

    for label in detected_set:
        if label in LABEL_INDEX:
            matched_topic_ids.update(LABEL_INDEX[label])
            matched_labels.add(label)

    if not matched_topic_ids:
        return {
            "tier": Tier.GREEN,
            "topics": [],
            "hierarchy": [],
            "labels_matched": [],
            "confidence": 0.0,
        }

    # Get matched topics
    matched_topics = [TOPICS_BY_ID[tid] for tid in matched_topic_ids]

    # Determine tier (highest tier value wins - higher number = more severe)
    tier = max(matched_topics, key=lambda t: t.tier.value).tier

    # Get topics for this tier only
    tier_topics = [t for t in matched_topics if t.tier == tier]

    # Build result
    return {
        "tier": tier,
        "tier_number": tier.value,
        "tier_name": tier.name,
        "topics": [t.id for t in tier_topics],
        "topic_names": [t.name for t in tier_topics],
        "hierarchy": [get_topic_hierarchy_display(t.id) for t in tier_topics],
        "labels_matched": list(matched_labels),
        "confidence": len(matched_labels) / len(detected_set) if detected_set else 0.0,
        "models_required": list(
            set(t.requires_model for t in tier_topics if t.requires_model)
        ),
    }


# =============================================================================
# Dashboard Display Functions
# =============================================================================


def get_taxonomy_tree() -> Dict[str, Any]:
    """
    Get complete taxonomy as nested tree for dashboard display.

    Returns:
        Hierarchical tree structure with all topics
    """

    def build_tree(
        topic_id: str, visited: Optional[Set[str]] = None
    ) -> Optional[Dict[str, Any]]:
        if visited is None:
            visited = set()

        if topic_id in visited:
            return None  # Avoid cycles

        visited.add(topic_id)
        topic = TOPICS_BY_ID[topic_id]

        # Find children
        children = [
            build_tree(child_id, visited.copy())
            for t in ALL_TOPICS
            if topic_id in t.parent_ids
            for child_id in [t.id]
        ]
        children = [c for c in children if c is not None]

        return {
            "id": topic.id,
            "name": topic.name,
            "tier": topic.tier.value,
            "tier_name": topic.tier.name,
            "emoji": topic.emoji,
            "color": topic.dashboard_color,
            "description": topic.description,
            "label_count": len(topic.labels),
            "children": children,
        }

    # Build tree from root topics (level 1)
    roots = [t for t in ALL_TOPICS if t.level == 1]
    return {
        "roots": [build_tree(t.id) for t in roots],
        "total_topics": len(ALL_TOPICS),
        "total_labels": len(LABEL_INDEX),
        "tiers": {
            "green": len(TIER_1_TOPICS),
            "yellow": len(TIER_2_TOPICS),
            "orange": len(TIER_3_TOPICS),
            "red": len(TIER_4_TOPICS),
        },
    }


def get_supervisor_dashboard_config() -> Dict[str, Any]:
    """
    Get dashboard configuration for supervisor topic management.

    Returns topic tree with toggle controls and tier indicators.
    """
    tree = get_taxonomy_tree()

    return {
        "taxonomy": tree,
        "tier_colors": {
            "GREEN": "#10B981",
            "YELLOW": "#F59E0B",
            "ORANGE": "#F97316",
            "RED": "#EF4444",
        },
        "tier_descriptions": {
            "GREEN": "Always allowed - Academic and safe topics",
            "YELLOW": "Needs approval - Sensitive topics requiring parent review",
            "ORANGE": "Requires supervision - Mental health concerns",
            "RED": "Auto-blocked - Crisis situations",
        },
        "toggle_options": {
            "allow_tier_1": True,  # Always true
            "allow_tier_2": False,  # Parent can toggle
            "allow_tier_3": False,  # Parent can toggle
            "allow_tier_4": False,  # Always false (auto-block)
        },
    }


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    print("ModAI Topic Taxonomy")
    print("=" * 80)
    print()

    # Example 1: Classify by labels
    print("Example 1: Algebra homework")
    result = classify_by_labels(["algebra", "homework", "help", "solve for x"])
    print(f"Tier: {result['tier_name']} ({result['tier_number']})")
    print(f"Topics: {result['topic_names']}")
    print(f"Hierarchy: {result['hierarchy']}")
    print(f"Labels matched: {result['labels_matched']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print()

    # Example 2: Crisis detection
    print("Example 2: Crisis situation")
    result = classify_by_labels(["suicide", "want to die", "kill myself"])
    print(f"Tier: {result['tier_name']} ({result['tier_number']})")
    print(f"Topics: {result['topic_names']}")
    print(f"Hierarchy: {result['hierarchy']}")
    print(f"Models required: {result['models_required']}")
    print()

    # Example 3: Get taxonomy stats
    print("Example 3: Taxonomy Statistics")
    tree = get_taxonomy_tree()
    print(f"Total topics: {tree['total_topics']}")
    print(f"Total labels: {tree['total_labels']}")
    print(f"Tier distribution:")
    for tier, count in tree["tiers"].items():
        print(f"  {tier}: {count} topics")
    print()

    # Example 4: Topic with multiple parents
    print("Example 4: School Anxiety (multiple parents)")
    hierarchy1 = get_topic_hierarchy_display("school_anxiety")
    print(f"Hierarchy: {hierarchy1}")
    print(f"Parents: {TOPICS_BY_ID['school_anxiety'].parent_ids}")
