"""
Test script for Topic Classifier Service
"""
import asyncio
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

async def test_classifier():
    """Test the classifier with sample messages"""
    from app.application.classifier import TopicClassifier
    from app.config import settings
    
    print("=" * 80)
    print("TOPIC CLASSIFIER TEST")
    print("=" * 80)
    print(f"Model: {settings.GEMINI_MODEL}")
    print(f"Cache: {'enabled' if settings.ENABLE_CACHING else 'disabled'}")
    print()
    
    # Initialize classifier
    print("Initializing classifier...")
    classifier = TopicClassifier()
    print("✅ Classifier initialized")
    print()
    
    # Test messages across all tiers
    test_messages = [
        {
            "message": "I need help with my algebra homework",
            "expected_tier": 1,
            "description": "Tier 1 (GREEN) - Academic"
        },
        {
            "message": "I'm worried about my body image and what people think",
            "expected_tier": 2,
            "description": "Tier 2 (YELLOW) - Sensitive"
        },
        {
            "message": "I feel so anxious and depressed all the time",
            "expected_tier": 3,
            "description": "Tier 3 (ORANGE) - Mental Health"
        },
        {
            "message": "I want to hurt myself, I can't take it anymore",
            "expected_tier": 4,
            "description": "Tier 4 (RED) - Crisis"
        }
    ]
    
    # Run tests
    for i, test in enumerate(test_messages, 1):
        print(f"Test {i}: {test['description']}")
        print(f"Message: \"{test['message']}\"")
        print()
        
        try:
            result = await classifier.classify(test['message'])
            
            print(f"✅ Tier: {result.tier} ({result.tier_name})")
            print(f"   Action: {result.action}")
            print(f"   Topics: {len(result.topics)}")
            
            for topic in result.topics[:3]:  # Show top 3
                print(f"   - {topic.topic_name} (Tier {topic.tier}, confidence: {topic.confidence:.2f})")
                print(f"     Labels: {', '.join(topic.labels_matched[:5])}")
            
            print(f"   Processing: {result.processing_time_ms:.1f}ms")
            print(f"   Cached: {result.cached}")
            
            # Check expected tier
            if result.tier == test['expected_tier']:
                print(f"   ✅ Expected tier matched!")
            else:
                print(f"   ⚠️ Expected tier {test['expected_tier']}, got {result.tier}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print()
        print("-" * 80)
        print()
    
    # Test taxonomy
    print("Testing taxonomy...")
    taxonomy = classifier.get_taxonomy()
    print(f"✅ Total topics: {taxonomy['total_topics']}")
    print(f"   Total labels: {taxonomy['total_labels']}")
    print(f"   Tier breakdown:")
    for tier, count in sorted(taxonomy['tiers'].items()):
        tier_name = {1: "GREEN", 2: "YELLOW", 3: "ORANGE", 4: "RED"}[tier]
        print(f"   - Tier {tier} ({tier_name}): {count} topics")
    
    print()
    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ ERROR: GOOGLE_API_KEY environment variable not set")
        print("Please create a .env file with your Google API key")
        sys.exit(1)
    
    asyncio.run(test_classifier())
