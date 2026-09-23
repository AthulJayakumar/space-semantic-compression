"""backend.app

Plain-English purpose: FastAPI web service and application orchestration.

This file is part of the public Space Semantic Compression research demo.
It is documented at module level so researchers, supervisors, and non-specialist
readers can understand where the file fits before reading implementation details.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import get_settings
from backend.logging_config import configure_logging
from backend.routes.compress import router as compress_router
from backend.routes.research import router as research_router
from backend.routes.reconstruct import router as reconstruct_router
from src.release import MODEL_ID, MODEL_STATUS, PROJECT_VERSION

settings = get_settings()
configure_logging(settings.log_level)

app = FastAPI(
    title=settings.app_name,
    version=PROJECT_VERSION,
    description="AI semantic token transmission for low-bandwidth environments.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(compress_router)
app.include_router(research_router)
app.include_router(reconstruct_router)


@app.get("/health")
async def health() -> dict[str, str | bool]:
    return {
        "status": "ok",
        "version": PROJECT_VERSION,
        "model_id": MODEL_ID,
        "model_status": MODEL_STATUS,
        "checkpoint_present": settings.checkpoint_path.is_file(),
    }
