"""
OpenEvals API

FastAPI backend for the OpenEvals evaluation framework.
Provides REST API and WebSocket endpoints for running benchmarks,
managing models, and viewing results.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

from .models import init_db
from .api.v1 import api_router

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("openevals.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Initializing database...")
    init_db()
    logger.info("Database initialized")
    yield
    logger.info("Shutting down...")


def custom_openapi():
    """Custom OpenAPI schema for better documentation."""
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="OpenEvals API",
        version="1.0.0",
        description="""
## OpenEvals - LLM Evaluation Framework

RESTful API for evaluating and benchmarking large language models.

### Features

- **Benchmark Management**: Create, run, and monitor evaluation runs
- **Model Configuration**: Register and configure models for evaluation
- **Task Selection**: Choose from 11+ standardized benchmarks
- **Real-time Updates**: WebSocket support for live progress
- **Results Analysis**: Query and compare benchmark results
- **Leaderboard**: Aggregated model rankings

### Supported Models

| Family | Sizes |
|--------|-------|
| Gemma, Gemma 3 | 1B - 27B |
| Llama 3, 3.1, 3.2 | 1B - 405B |
| Mistral, Mixtral | 7B - 8x22B |
| Qwen 2, 2.5 | 0.5B - 72B |
| DeepSeek, DeepSeek-R1 | 1.5B - 671B |
| Phi-3, OLMo | Various |

### Supported Benchmarks

MMLU, GSM8K, MATH, HumanEval, TruthfulQA, HellaSwag, ARC, WinoGrande, GPQA, IFEval, BBH

### Authentication

Currently, no authentication is required. For production deployments,
implement API key or OAuth2 authentication.

### Rate Limiting

No rate limiting is enforced. For production, consider implementing
rate limiting based on your infrastructure requirements.
        """,
        routes=app.routes,
        tags=[
            {
                "name": "benchmarks",
                "description": "Create and manage benchmark evaluation runs",
            },
            {
                "name": "models",
                "description": "Configure and manage model definitions",
            },
            {
                "name": "tasks",
                "description": "List available benchmark tasks",
            },
            {
                "name": "websocket",
                "description": "Real-time benchmark progress updates",
            },
        ],
    )

    logo_url = "https://raw.githubusercontent.com/heilcheng/openevals/main/logo.png"
    openapi_schema["info"]["x-logo"] = {"url": logo_url}

    openapi_schema["info"]["contact"] = {
        "name": "OpenEvals",
        "url": "https://github.com/heilcheng/openevals",
    }

    openapi_schema["info"]["license"] = {
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    }

    app.openapi_schema = openapi_schema
    return app.openapi_schema


# Create FastAPI app
app = FastAPI(
    title="OpenEvals API",
    description="API for LLM evaluation and benchmarking",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

app.openapi = custom_openapi

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api/v1")


@app.get("/", tags=["root"])
async def root():
    """API root endpoint."""
    return {
        "name": "OpenEvals API",
        "version": "1.0.0",
        "description": "LLM Evaluation Framework",
        "docs": "/api/docs",
        "health": "/health",
    }


@app.get("/health", tags=["root"])
async def health_check():
    """Health check endpoint for monitoring."""
    return {"status": "healthy", "service": "openevals-api"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
