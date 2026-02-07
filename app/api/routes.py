"""
FastAPI routes for Topic Classifier Service
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
import redis

from ..domain.models import (
    ClassificationRequest,
    ClassificationResponse,
    TaxonomyResponse,
    HealthResponse,
)
from ..application.classifier import get_classifier, TopicClassifier
from ..config import settings

# Create API router
router = APIRouter(prefix="/api/v1", tags=["topic-classifier"])


# Async dependency for getting classifier
async def get_classifier_dependency(timeout: float = 60.0) -> TopicClassifier:
    """
    FastAPI dependency for getting the classifier instance.
    Waits up to 60 seconds for classifier initialization.

    Args:
        timeout: Maximum seconds to wait for initialization

    Returns:
        Initialized TopicClassifier instance

    Raises:
        HTTPException: If classifier fails to initialize or times out
    """
    return await get_classifier(timeout=timeout)


@router.post("/classify", response_model=ClassificationResponse)
async def classify_message(
    request: ClassificationRequest,
    classifier: TopicClassifier = Depends(get_classifier_dependency),
) -> ClassificationResponse:
    """
    Classify a message into topics

    **Request:**
    ```json
    {
        "message": "I need help with algebra homework",
        "teen_id": "optional-teen-id",
        "context": {}
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
    """
    try:
        result = await classifier.classify(
            message=request.message, teen_id=request.teen_id
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")


@router.get("/taxonomy", response_model=TaxonomyResponse)
async def get_taxonomy(
    classifier: TopicClassifier = Depends(get_classifier_dependency),
) -> TaxonomyResponse:
    """
    Get the complete topic taxonomy

    **Response:**
    ```json
    {
        "total_topics": 59,
        "total_labels": 345,
        "tiers": {
            "1": 20,
            "2": 12,
            "3": 15,
            "4": 12
        },
        "topics": [...]
    }
    ```
    """
    try:
        taxonomy = classifier.get_taxonomy()
        return TaxonomyResponse(**taxonomy)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve taxonomy: {str(e)}"
        )


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint - responds immediately without waiting for classifier

    **Response:**
    ```json
    {
        "status": "healthy",
        "service": "topic-classifier-service",
        "version": "1.0.0",
        "gemini_model": "gemini-2.0-flash-exp",
        "redis_connected": true,
        "classifier_status": "ready"
    }
    ```
    """
    from ..application.classifier import _classifier_ready, _classifier_error, _classifier

    # Check Redis connection
    redis_connected = False
    try:
        r = redis.Redis(
            host=settings.REDIS_HOST, port=settings.REDIS_PORT, db=settings.REDIS_DB
        )
        r.ping()
        redis_connected = True
    except:
        pass

    # Check classifier status (non-blocking)
    classifier_status = "initializing"
    model_loaded = False
    model_path = None

    if _classifier_ready.is_set():
        if _classifier_error:
            classifier_status = "failed"
        else:
            classifier_status = "ready"
            model_loaded = True
            if _classifier:
                model_path = getattr(_classifier, "model_path", None)

    return HealthResponse(
        status="healthy",
        service=settings.SERVICE_NAME,
        version="1.0.0",
        gemini_model=settings.GEMINI_MODEL,
        redis_connected=redis_connected,
        classifier_status=classifier_status,
        model_loaded=model_loaded,
        model_path=model_path,
    )


@router.get("/inspect")
async def inspect_model(classifier: TopicClassifier = Depends(get_classifier_dependency)):
    """Return classifier/ONNX model diagnostics and a small sample inference"""
    try:
        taxonomy = classifier.get_taxonomy()
        # Collect a small label list for a sample classification
        sample_labels = []
        for t in taxonomy.get("topics", []):
            for l in t.get("labels", []):
                if l not in sample_labels:
                    sample_labels.append(l)
                if len(sample_labels) >= 10:
                    break
            if len(sample_labels) >= 10:
                break

        model_info = {
            "method": classifier.method,
            "model_path": getattr(classifier, "model_path", None),
            "tokenizer_type": getattr(classifier, "tokenizer_type", None),
            "tokenizer_is_fast": getattr(classifier, "tokenizer_is_fast", None),
            "tokenizer_input_names": getattr(classifier, "tokenizer_input_names", None),
            "topics_loaded": taxonomy.get("total_topics"),
        }
        # Run a tiny inference if the classifier is available
        sample_output = None
        try:
            sample_output = None
            if getattr(classifier, "classifier", None) is not None:
                sample_output = classifier.classifier(
                    "I am worried about my friend",
                    sample_labels,
                    multi_label=True,
                )
        except Exception as e:
            sample_output = {"error": str(e)}

        return {
            "model_info": model_info,
            "sample_output": sample_output,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/")
async def root() -> Dict[str, Any]:
    """
    Root endpoint with service information
    """
    return {
        "service": settings.SERVICE_NAME,
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "classify": "/api/v1/classify",
            "taxonomy": "/api/v1/taxonomy",
            "health": "/api/v1/health",
            "docs": "/docs",
        },
        "model": settings.GEMINI_MODEL,
        "description": "Topic classification service using Google Gemini Flash 2.0",
    }
