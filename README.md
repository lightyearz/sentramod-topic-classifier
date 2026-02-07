# Topic Classifier Service

AI-powered topic classification service using Google Gemini Flash 2.0 for the ModAI teen safety platform.

## Overview

This service classifies messages into 59 hierarchical topics across 4 safety tiers:
- **Tier 1 (GREEN)**: Academic, safe topics - allowed by default
- **Tier 2 (YELLOW)**: Sensitive topics - require supervisor approval
- **Tier 3 (ORANGE)**: Mental health - require monitoring and alerts
- **Tier 4 (RED)**: Crisis situations - blocked immediately, crisis response triggered

## Features

- **Gemini Flash 2.0**: Lightweight, fast Google AI model for classification
- **Hierarchical Taxonomy**: 59 topics with 345 natural-language labels
- **Redis Caching**: Fast responses for repeated messages
- **Multi-tier Safety**: 4-tier classification system
- **RESTful API**: Easy integration with FastAPI

## Quick Start

### 1. Install Dependencies

```bash
cd services/topic-classifier-service
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

### 3. Run the Service

```bash
python -m app.main
```

Or with uvicorn:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8007 --reload
```

The service will be available at:
- API: http://localhost:8007
- Docs: http://localhost:8007/docs
- Health: http://localhost:8007/api/v1/health

## API Endpoints

### POST /api/v1/classify

Classify a message into topics.

**Request:**
```json
{
  "message": "I need help with algebra homework",
  "teen_id": "optional-teen-id"
}
```

**Response:**
```json
{
  "message": "I need help with algebra homework",
  "tier": 1,
  "tier_name": "GREEN",
  "topics": [
    {
      "topic_id": "algebra",
      "topic_name": "Algebra",
      "tier": 1,
      "confidence": 0.9,
      "labels_matched": ["algebra", "homework"],
      "hierarchy": "Academic > Mathematics > Algebra"
    }
  ],
  "action": "allow",
  "model_used": "gemini-2.0-flash-exp",
  "processing_time_ms": 145.2,
  "cached": false
}
```

### GET /api/v1/taxonomy

Get the complete topic taxonomy (59 topics, 345 labels).

### GET /api/v1/health

Health check endpoint.

## Architecture

### File Structure

```
topic-classifier-service/
├── app/
│   ├── main.py                 # FastAPI app
│   ├── config.py               # Configuration
│   ├── api/
│   │   ├── routes.py           # Classification API
│   │   └── chat_routes.py      # OpenAI-compatible chat gateway
│   ├── application/
│   │   └── classifier.py       # Gemini classifier
│   └── domain/
│       └── models.py           # Pydantic models
├── requirements.txt
├── .env.example
└── README.md
```

### Safety Architecture: Separation of Concerns

ModAI uses a **two-layer safety system** with clear separation:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MODAI SAFETY ARCHITECTURE                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────┐                                    │
│  │        LAYER 1: TOPIC CLASSIFIER    │  ← WHAT can be discussed          │
│  │        (Content Restriction)        │                                    │
│  ├─────────────────────────────────────┤                                    │
│  │  • 4-Tier Safety System             │                                    │
│  │  • Topic classification (59 topics) │                                    │
│  │  • Block Tier 4 (crisis/danger)     │                                    │
│  │  • Request approval for Tier 2/3    │                                    │
│  │  • Parent notifications             │                                    │
│  │  • Crisis resource routing          │                                    │
│  └─────────────────────────────────────┘                                    │
│                    │                                                        │
│                    ▼ (if message allowed)                                   │
│  ┌─────────────────────────────────────┐                                    │
│  │        LAYER 2: SYSTEM PROMPT       │  ← HOW to respond/teach            │
│  │        (Educational Philosophy)     │                                    │
│  ├─────────────────────────────────────┤                                    │
│  │  • Guide, don't give direct answers │                                    │
│  │  • Ask clarifying questions         │                                    │
│  │  • Break problems into steps        │                                    │
│  │  • Explain "why" not just "what"    │                                    │
│  │  • Celebrate when they figure it out│                                    │
│  │  • Age-appropriate communication    │                                    │
│  │  • NO TOPIC RESTRICTIONS HERE       │                                    │
│  └─────────────────────────────────────┘                                    │
│                    │                                                        │
│                    ▼                                                        │
│             ┌──────────────┐                                                │
│             │  AI Provider │ (Gemini/Claude)                                │
│             └──────────────┘                                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Why This Separation Matters:**

| Layer                | Responsibility                             | Does NOT Do     |
| -------------------- | ------------------------------------------ | --------------- |
| **Topic Classifier** | Block dangerous content, require approvals | Teach or tutor  |
| **System Prompt**    | Guide learning approach, educational tone  | Restrict topics |

This ensures:
1. **Clear responsibility** - Topic Classifier handles safety, System Prompt handles pedagogy
2. **No double-restrictions** - Topics aren't blocked twice
3. **Flexibility** - Can update teaching style without affecting safety rules
4. **Auditability** - Safety decisions are logged separately from teaching interactions

## Integration with TOPIC_TAXONOMY

This service integrates with `/models/TOPIC_TAXONOMY.py` which contains:
- 59 topics across 4 tiers
- 345 natural-language labels
- Hierarchical topic structure
- Set-based classification logic

## Caching

Redis caching is enabled by default for performance:
- Cache TTL: 1 hour (configurable)
- Cache key: MD5 hash of message
- Cache DB: 3 (configurable)

## Configuration

All settings can be configured via environment variables:

| Variable               | Default              | Description               |
| ---------------------- | -------------------- | ------------------------- |
| `GOOGLE_API_KEY`       | -                    | Google API key (required) |
| `SERVICE_PORT`         | 8007                 | Service port              |
| `GEMINI_MODEL`         | gemini-2.0-flash-exp | Gemini model to use       |
| `REDIS_HOST`           | localhost            | Redis host                |
| `REDIS_PORT`           | 6379                 | Redis port                |
| `ENABLE_CACHING`       | True                 | Enable/disable caching    |
| `CONFIDENCE_THRESHOLD` | 0.6                  | Min confidence for topics |

## Testing

Test the service with sample messages:

```bash
# Start the service
python -m app.main

# In another terminal, test with curl
curl -X POST http://localhost:8007/api/v1/classify \
  -H "Content-Type: application/json" \
  -d '{"message": "I need help with algebra homework"}'
```

## Performance

- **Latency**: ~100-200ms per message (with Gemini Flash 2.0)
- **Caching**: ~5-10ms for cached messages
- **Throughput**: ~50-100 requests/second per instance
- **Cost**: $0 (uses user's Google API key)

## Next Steps

1. Integrate with Message Service (Port 8001)
2. Add database integration for topic permissions
3. Implement approval queue for Tier 2
4. Add crisis alert system for Tier 3/4
5. Build dashboard UI for topic management

## License

Proprietary - ModAI Platform
