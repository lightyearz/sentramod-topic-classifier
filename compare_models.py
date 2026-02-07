"""
Compare Gemini API vs ONNX/BART local classification

This script tests the same messages with both approaches and compares:
- Latency
- Cost
- Topics detected
- Tier classification
- Accuracy

Usage:
    python services/topic-classifier-service/compare_models.py
"""

import os
import sys
import time
import json
import asyncio
from typing import List, Dict, Any

# Add app to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

# Test messages covering all tiers
TEST_MESSAGES = [
    {
        "message": "I need help with my algebra homework",
        "expected_tier": 1,
        "expected_topics": ["algebra", "mathematics", "homework"]
    },
    {
        "message": "I want to ask my crush to the school dance",
        "expected_tier": 2,
        "expected_topics": ["dating", "relationships", "first crush"]
    },
    {
        "message": "I feel really anxious about my exams and can't sleep",
        "expected_tier": 3,
        "expected_topics": ["anxiety", "stress", "sleep problems"]
    },
    {
        "message": "I want to hurt myself, I can't take it anymore",
        "expected_tier": 4,
        "expected_topics": ["self-harm", "suicidal ideation", "crisis"]
    },
    {
        "message": "Can you help me write a creative story about a dragon?",
        "expected_tier": 1,
        "expected_topics": ["creative writing", "storytelling", "hobbies"]
    },
    {
        "message": "My friend is being cyberbullied on Instagram",
        "expected_tier": 3,
        "expected_topics": ["cyberbullying", "social media", "peer relationships"]
    },
    {
        "message": "I think I might be depressed, I don't enjoy anything anymore",
        "expected_tier": 3,
        "expected_topics": ["depression", "mental health", "lack of motivation"]
    },
    {
        "message": "What's the best way to study for a history test?",
        "expected_tier": 1,
        "expected_topics": ["study skills", "history", "academic performance"]
    },
]


async def test_onnx_bart():
    """Test ONNX/BART local classification"""
    print("=" * 80)
    print("TESTING ONNX/BART LOCAL CLASSIFICATION")
    print("=" * 80)

    # Set environment to use ONNX
    os.environ['CLASSIFICATION_METHOD'] = 'onnx'

    # Import after setting env
    from app.application.classifier import get_classifier
    from app.config import settings

    # Clear any existing classifier
    import app.application.classifier as clf_module
    clf_module._classifier = None

    # Disable Redis caching for accurate timing
    settings.ENABLE_CACHING = False

    classifier = get_classifier()
    print(f"✅ Classifier loaded: {classifier.method}")
    print(f"   Model path: {getattr(classifier, 'model_path', 'N/A')}")
    print()

    results = []
    total_time = 0

    for i, test in enumerate(TEST_MESSAGES, 1):
        print(f"Test {i}/{len(TEST_MESSAGES)}: {test['message'][:60]}...")

        start = time.time()
        result = await classifier.classify(test['message'])
        elapsed = (time.time() - start) * 1000
        total_time += elapsed

        result_dict = result.model_dump() if hasattr(result, 'model_dump') else result.dict()

        results.append({
            "message": test['message'],
            "expected_tier": test['expected_tier'],
            "actual_tier": result_dict['tier'],
            "tier_match": result_dict['tier'] == test['expected_tier'],
            "topics": [t['topic_name'] for t in result_dict['topics']],
            "action": result_dict['action'],
            "latency_ms": elapsed,
            "model_used": result_dict['model_used'],
            "cached": result_dict.get('cached', False)
        })

        print(f"   Tier: {result_dict['tier']} ({result_dict['tier_name']}) - Expected: {test['expected_tier']} ✅" if result_dict['tier'] == test['expected_tier'] else f"   Tier: {result_dict['tier']} ({result_dict['tier_name']}) - Expected: {test['expected_tier']} ❌")
        print(f"   Topics: {', '.join([t['topic_name'] for t in result_dict['topics'][:3]])}")
        print(f"   Latency: {elapsed:.1f}ms")
        print(f"   Action: {result_dict['action']}")
        print()

    avg_latency = total_time / len(TEST_MESSAGES)
    tier_accuracy = sum(1 for r in results if r['tier_match']) / len(results) * 100

    print("-" * 80)
    print(f"ONNX/BART SUMMARY:")
    print(f"   Total time: {total_time:.1f}ms")
    print(f"   Average latency: {avg_latency:.1f}ms")
    print(f"   Tier accuracy: {tier_accuracy:.1f}%")
    print(f"   Cost per classification: $0 (local)")
    print(f"   Cost for 1000 users/month: $0")
    print(f"   Network required: No")
    print(f"   Offline capable: Yes")
    print("-" * 80)
    print()

    return {
        "method": "ONNX/BART",
        "results": results,
        "total_time_ms": total_time,
        "avg_latency_ms": avg_latency,
        "tier_accuracy": tier_accuracy,
        "cost_per_classification": 0,
        "cost_1000_users_month": 0,
        "network_required": False,
        "offline_capable": True
    }


async def test_gemini():
    """Test Gemini API classification (simulated based on session-service implementation)"""
    print("=" * 80)
    print("TESTING GEMINI API CLASSIFICATION")
    print("=" * 80)
    print("⚠️  NOTE: This is a simulated test based on session-service's Gemini usage")
    print("   The actual Gemini implementation is in session-service, not topic-classifier-service")
    print()

    try:
        import google.generativeai as genai
        from app.config import settings

        if not settings.GOOGLE_API_KEY:
            print("❌ GOOGLE_API_KEY not set. Cannot test Gemini.")
            return None

        genai.configure(api_key=settings.GOOGLE_API_KEY)
        model = genai.GenerativeModel(settings.GEMINI_MODEL)

        print(f"✅ Gemini model loaded: {settings.GEMINI_MODEL}")
        print()

        results = []
        total_time = 0

        for i, test in enumerate(TEST_MESSAGES, 1):
            print(f"Test {i}/{len(TEST_MESSAGES)}: {test['message'][:60]}...")

            prompt = f"""Classify this teen message into safety tiers:

Message: "{test['message']}"

Respond with JSON:
{{
  "tier": 1-4,
  "tier_name": "GREEN/YELLOW/ORANGE/RED",
  "topics": ["topic1", "topic2"],
  "action": "allow/require_approval/alert/block"
}}

Tiers:
- 1 (GREEN): Academic, hobbies - always allowed
- 2 (YELLOW): Dating, social media - needs approval
- 3 (ORANGE): Mental health, anxiety - requires supervision
- 4 (RED): Self-harm, suicide, violence - block immediately"""

            start = time.time()
            try:
                response = model.generate_content(prompt)
                elapsed = (time.time() - start) * 1000
                total_time += elapsed

                # Parse JSON from response
                text = response.text.strip()
                if text.startswith('```json'):
                    text = text[7:]
                if text.endswith('```'):
                    text = text[:-3]
                result = json.loads(text.strip())

                results.append({
                    "message": test['message'],
                    "expected_tier": test['expected_tier'],
                    "actual_tier": result['tier'],
                    "tier_match": result['tier'] == test['expected_tier'],
                    "topics": result.get('topics', []),
                    "action": result['action'],
                    "latency_ms": elapsed,
                    "model_used": settings.GEMINI_MODEL,
                    "cached": False
                })

                print(f"   Tier: {result['tier']} ({result['tier_name']}) - Expected: {test['expected_tier']} ✅" if result['tier'] == test['expected_tier'] else f"   Tier: {result['tier']} ({result['tier_name']}) - Expected: {test['expected_tier']} ❌")
                print(f"   Topics: {', '.join(result.get('topics', [])[:3])}")
                print(f"   Latency: {elapsed:.1f}ms")
                print(f"   Action: {result['action']}")
                print()

            except Exception as e:
                print(f"   ❌ Error: {e}")
                elapsed = (time.time() - start) * 1000
                total_time += elapsed
                results.append({
                    "message": test['message'],
                    "error": str(e),
                    "latency_ms": elapsed
                })
                print()

        successful_results = [r for r in results if 'tier_match' in r]
        if successful_results:
            avg_latency = total_time / len(TEST_MESSAGES)
            tier_accuracy = sum(1 for r in successful_results if r['tier_match']) / len(successful_results) * 100

            # Cost calculation (Gemini Flash 2.0 pricing)
            # Input: $0.075 per 1M tokens, Output: $0.30 per 1M tokens
            # Estimate ~100 input tokens, ~50 output tokens per classification
            cost_per_classification = (100 * 0.075 + 50 * 0.30) / 1_000_000
            # 1000 users * 400 messages/month * cost_per_classification
            cost_1000_users = 1000 * 400 * cost_per_classification

            print("-" * 80)
            print(f"GEMINI SUMMARY:")
            print(f"   Total time: {total_time:.1f}ms")
            print(f"   Average latency: {avg_latency:.1f}ms")
            print(f"   Tier accuracy: {tier_accuracy:.1f}%")
            print(f"   Cost per classification: ${cost_per_classification:.6f}")
            print(f"   Cost for 1000 users/month: ${cost_1000_users:.2f}")
            print(f"   Network required: Yes")
            print(f"   Offline capable: No")
            print("-" * 80)
            print()

            return {
                "method": "Gemini Flash 2.0",
                "results": results,
                "total_time_ms": total_time,
                "avg_latency_ms": avg_latency,
                "tier_accuracy": tier_accuracy,
                "cost_per_classification": cost_per_classification,
                "cost_1000_users_month": cost_1000_users,
                "network_required": True,
                "offline_capable": False
            }
        else:
            print("❌ No successful Gemini classifications")
            return None

    except ImportError:
        print("❌ google-generativeai not installed. Cannot test Gemini.")
        return None
    except Exception as e:
        print(f"❌ Gemini test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def print_comparison(onnx_results, gemini_results):
    """Print side-by-side comparison"""
    print()
    print("=" * 80)
    print("COMPARISON: GEMINI vs ONNX/BART")
    print("=" * 80)
    print()

    if not gemini_results:
        print("⚠️  Gemini results not available for comparison")
        print()
        print("ONNX/BART is the only option currently configured.")
        print()
        print("If you want to test Gemini:")
        print("1. Ensure GOOGLE_API_KEY is set in .env")
        print("2. Install google-generativeai: pip install google-generativeai")
        print("3. Run this script again")
        return

    # Performance comparison
    print("📊 PERFORMANCE:")
    print(f"   {'Metric':<30} {'Gemini':<20} {'ONNX/BART':<20} {'Winner':<10}")
    print(f"   {'-'*30} {'-'*20} {'-'*20} {'-'*10}")

    # Latency
    gemini_latency = gemini_results['avg_latency_ms']
    onnx_latency = onnx_results['avg_latency_ms']
    latency_winner = "ONNX/BART" if onnx_latency < gemini_latency else "Gemini"
    latency_improvement = ((gemini_latency - onnx_latency) / gemini_latency * 100) if gemini_latency > 0 else 0
    print(f"   {'Average Latency':<30} {gemini_latency:>18.1f}ms {onnx_latency:>18.1f}ms {'✅ ' + latency_winner:<10}")
    if latency_improvement > 0:
        print(f"   {'  → Improvement':<30} {'':<20} {latency_improvement:>17.1f}% {'faster':<10}")

    # Accuracy
    gemini_accuracy = gemini_results['tier_accuracy']
    onnx_accuracy = onnx_results['tier_accuracy']
    accuracy_winner = "ONNX/BART" if onnx_accuracy >= gemini_accuracy else "Gemini"
    print(f"   {'Tier Accuracy':<30} {gemini_accuracy:>18.1f}% {onnx_accuracy:>18.1f}% {'✅ ' + accuracy_winner:<10}")

    print()

    # Cost comparison
    print("💰 COST:")
    print(f"   {'Metric':<30} {'Gemini':<20} {'ONNX/BART':<20} {'Savings':<10}")
    print(f"   {'-'*30} {'-'*20} {'-'*20} {'-'*10}")

    gemini_cost_per = gemini_results['cost_per_classification']
    onnx_cost_per = onnx_results['cost_per_classification']
    print(f"   {'Cost per classification':<30} ${gemini_cost_per:>17.6f} ${onnx_cost_per:>17.2f} {'✅ 100%':<10}")

    gemini_cost_1000 = gemini_results['cost_1000_users_month']
    onnx_cost_1000 = onnx_results['cost_1000_users_month']
    monthly_savings = gemini_cost_1000 - onnx_cost_1000
    print(f"   {'Cost (1000 users/month)':<30} ${gemini_cost_1000:>17.2f} ${onnx_cost_1000:>17.2f} {'✅ $' + f'{monthly_savings:.2f}':<10}")

    yearly_savings = monthly_savings * 12
    print(f"   {'Annual Savings (1000 users)':<30} {'':<20} {'':<20} {'$' + f'{yearly_savings:.2f}':<10}")

    print()

    # Other factors
    print("🔧 OTHER FACTORS:")
    print(f"   {'Factor':<30} {'Gemini':<20} {'ONNX/BART':<20}")
    print(f"   {'-'*30} {'-'*20} {'-'*20}")
    print(f"   {'Network Required':<30} {'Yes ❌':<20} {'No ✅':<20}")
    print(f"   {'Offline Capable':<30} {'No ❌':<20} {'Yes ✅':<20}")
    print(f"   {'Rate Limits':<30} {'Yes (60/min) ❌':<20} {'None ✅':<20}")
    print(f"   {'Privacy':<30} {'External API ❌':<20} {'Local ✅':<20}")
    print(f"   {'Setup Complexity':<30} {'API Key only ✅':<20} {'Model download ❌':<20}")

    print()

    # Recommendation
    print("💡 RECOMMENDATION:")
    print()

    if onnx_accuracy >= 85 and onnx_latency < gemini_latency * 1.5:
        print("   ✅ USE ONNX/BART:")
        print("      - Similar or better accuracy")
        print(f"      - {latency_improvement:.0f}% faster")
        print(f"      - ${monthly_savings:.2f}/month savings at 1000 users")
        print("      - Works offline")
        print("      - Better privacy (no external API calls)")
    elif gemini_accuracy > onnx_accuracy + 10:
        print("   ⚠️  HYBRID APPROACH RECOMMENDED:")
        print("      - Use ONNX/BART for most classifications (fast, free)")
        print("      - Fall back to Gemini for low-confidence cases (accurate)")
        print(f"      - Estimated savings: ${monthly_savings * 0.9:.2f}/month (90% ONNX, 10% Gemini)")
    else:
        print("   ✅ USE ONNX/BART (Current Implementation):")
        print("      - Comparable accuracy")
        print("      - Significantly lower cost")
        print("      - Better privacy and offline capability")

    print()


async def main():
    """Run comparison tests"""
    print()
    print("🔬 MODEL COMPARISON TEST")
    print("   Comparing Gemini API vs ONNX/BART local classification")
    print()

    # Test ONNX/BART
    onnx_results = await test_onnx_bart()

    # Test Gemini
    gemini_results = await test_gemini()

    # Print comparison
    print_comparison(onnx_results, gemini_results)

    # Save results
    output_file = "model_comparison_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            "onnx_bart": onnx_results,
            "gemini": gemini_results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2, default=str)

    print(f"📄 Detailed results saved to: {output_file}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
