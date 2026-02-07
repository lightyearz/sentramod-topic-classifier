"""
Topic Classifier Service
FastAPI microservice for topic classification using Google Gemini Flash 2.0
"""

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("🟢 main.py: Starting imports...")

from fastapi import FastAPI
logger.info("🟢 main.py: Imported FastAPI")

from fastapi.middleware.cors import CORSMiddleware
logger.info("🟢 main.py: Imported CORSMiddleware")

import uvicorn
logger.info("🟢 main.py: Imported uvicorn")

from .api.routes import router
logger.info("🟢 main.py: Imported router")

from .api.chat_routes import chat_router
logger.info("🟢 main.py: Imported chat_router (OpenAI-compatible gateway)")

from .api.dashboard import dashboard_router
logger.info("🟢 main.py: Imported dashboard_router (Test UI)")

from .application.classifier import get_classifier
logger.info("🟢 main.py: Imported get_classifier")

from .config import settings
logger.info("🟢 main.py: Imported settings")
logger.info("🟢 main.py: All imports complete!")

# Create FastAPI app
app = FastAPI(
    title="Topic Classifier Service",
    description="AI-powered topic classification for teen safety platform using Google Gemini Flash 2.0",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure based on environment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)

# Include OpenAI-compatible chat routes (Gateway functionality)
app.include_router(chat_router)

# Include Test Dashboard (accessible at /dashboard and /test)
app.include_router(dashboard_router)


@app.on_event("startup")
async def startup_event():
    """Initialize service on startup"""
    import asyncio
    from app.application.classifier import initialize_classifier

    logger.info("=" * 60)
    logger.info(f"🚀 Starting {settings.SERVICE_NAME}")
    logger.info(f"📊 Classification method: {settings.CLASSIFICATION_METHOD}")
    logger.info(f"📊 Using Gemini model: {settings.GEMINI_MODEL}")
    logger.info(f"🔌 Redis caching: {'enabled' if settings.ENABLE_CACHING else 'disabled'}")
    logger.info(f"🌐 Service running on http://{settings.HOST}:{settings.SERVICE_PORT}")
    logger.info(
        f"📚 API docs available at http://{settings.HOST}:{settings.SERVICE_PORT}/docs"
    )
    logger.info("=" * 60)

    # Start classifier initialization in background (non-blocking)
    # The model will load in a separate thread while the service accepts requests
    logger.info("⏳ Starting classifier initialization in background thread...")

    try:
        # Create background task and store reference to prevent garbage collection
        task = asyncio.create_task(initialize_classifier())

        # Add error handler for background task
        def handle_task_result(task):
            try:
                task.result()
            except Exception as e:
                logger.error(f"❌ Classifier initialization failed: {e}", exc_info=True)

        task.add_done_callback(handle_task_result)
        logger.info("📡 Service ready! (Classifier loading in background)")
    except Exception as e:
        logger.error(f"❌ Failed to create initialization task: {e}", exc_info=True)
        logger.info("📡 Service ready! (Classifier initialization FAILED TO START)")

    logger.info("=" * 60)
    logger.info("✅ Startup complete! Health checks will work immediately.")
    logger.info("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print(f"🛑 Shutting down {settings.SERVICE_NAME}")


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.SERVICE_PORT,
        reload=settings.DEBUG,
        log_level="info",
    )
