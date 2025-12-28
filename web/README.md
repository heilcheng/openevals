# Benchmark Suite Web Platform

A web-based interface for the LLM evaluation framework, designed for systematic benchmarking and comparison of large language models.

## Overview

This platform provides researchers with a streamlined workflow for:

- Configuring and executing benchmark evaluations across multiple models
- Monitoring evaluation progress in real-time via WebSocket connections
- Analyzing results through interactive visualizations
- Comparing model performance across tasks via aggregated leaderboards

The system integrates directly with the `gemma_benchmark` core library, supporting all registered model loaders and benchmark tasks.

## Architecture

```
web/
├── backend/           # FastAPI REST API + WebSocket server
│   ├── app/
│   │   ├── api/v1/    # Versioned API endpoints
│   │   ├── models/    # SQLAlchemy ORM models
│   │   ├── schemas/   # Pydantic request/response schemas
│   │   ├── services/  # Business logic layer
│   │   └── utils/     # Benchmark adapter and utilities
│   └── requirements.txt
│
└── frontend/          # Next.js 15 web application
    ├── src/
    │   ├── app/       # Page routes (App Router)
    │   ├── components/# React components
    │   ├── hooks/     # Custom React hooks
    │   ├── lib/       # API client and utilities
    │   └── stores/    # Zustand state management
    └── package.json
```

## Technical Stack

### Backend
- **FastAPI** - Async Python web framework
- **SQLAlchemy 2.0** - ORM with async support
- **SQLite / PostgreSQL** - Configurable database backend
- **WebSocket** - Real-time progress streaming

### Frontend
- **Next.js 15** - React framework with App Router
- **TypeScript** - Type-safe development
- **Tailwind CSS** - Utility-first styling
- **Recharts** - Data visualization
- **Zustand** - Lightweight state management

## Installation

### Prerequisites

- Python 3.10+
- Node.js 18+
- The `gemma_benchmark` package installed in the Python environment

### Backend Setup

```bash
cd web/backend
pip install -r requirements.txt
```

### Frontend Setup

```bash
cd web/frontend
npm install
```

## Running the Platform

### Start the Backend Server

```bash
cd web/backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API documentation is available at `http://localhost:8000/api/docs`.

### Start the Frontend Development Server

```bash
cd web/frontend
npm run dev
```

Access the web interface at `http://localhost:3000`.

## API Reference

### Benchmark Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/benchmarks` | List all benchmark runs |
| `POST` | `/api/v1/benchmarks` | Create and start a new benchmark run |
| `GET` | `/api/v1/benchmarks/{id}` | Retrieve benchmark details and results |
| `POST` | `/api/v1/benchmarks/{id}/cancel` | Cancel a running benchmark |
| `DELETE` | `/api/v1/benchmarks/{id}` | Delete a benchmark run |
| `GET` | `/api/v1/benchmarks/stats` | Aggregate statistics |
| `GET` | `/api/v1/benchmarks/leaderboard` | Model rankings by performance |

### Model Configuration Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/models` | List saved model configurations |
| `POST` | `/api/v1/models` | Create a new model configuration |
| `GET` | `/api/v1/models/types` | Available model types and defaults |
| `DELETE` | `/api/v1/models/{id}` | Delete a model configuration |

### Task Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/tasks` | List available benchmark tasks |
| `GET` | `/api/v1/tasks/{name}` | Task details and default configuration |

### WebSocket

| Endpoint | Description |
|----------|-------------|
| `ws://host/ws/benchmark/{run_id}` | Real-time progress updates for a benchmark run |

## Supported Benchmark Tasks

| Task | Description | Primary Metric |
|------|-------------|----------------|
| MMLU | Multitask language understanding (57 subjects) | Accuracy |
| GSM8K | Grade school math word problems | Accuracy |
| MATH | Competition mathematics (AMC, AIME, Olympiad) | Accuracy |
| HumanEval | Python code generation | Pass@k |
| TruthfulQA | Truthfulness evaluation | MC1/MC2 |
| HellaSwag | Commonsense reasoning | Accuracy |
| ARC | Science reasoning (Easy/Challenge) | Accuracy |
| WinoGrande | Winograd Schema Challenge | Accuracy |
| GPQA | Graduate-level science Q&A | Accuracy |
| IFEval | Instruction following | Strict/Loose Accuracy |
| BBH | BIG-Bench Hard reasoning | Accuracy |

Additional tasks can be registered through the `gemma_benchmark` task factory.

## Supported Model Types

| Type | Models | Sizes |
|------|--------|-------|
| `gemma` | Gemma, Gemma 2 | 2B, 7B, 9B, 27B |
| `gemma3` | Gemma 3 | 1B, 4B, 12B, 27B |
| `llama3` | Llama 3, 3.1, 3.2 | 1B - 405B |
| `mistral` | Mistral, Mixtral | 7B, 8x7B, 8x22B |
| `qwen2.5` | Qwen 2, Qwen 2.5 | 0.5B - 72B |
| `deepseek` | DeepSeek, DeepSeek-R1 | 1.5B - 671B |
| `phi3` | Phi-3 | Mini, Small, Medium |
| `olmo` | OLMo | 1B, 7B |
| `huggingface` | Any HuggingFace model | Custom |

## Configuration

### Environment Variables

**Backend** (`.env`):
```
DATABASE_URL=sqlite:///./benchmark.db
```

**Frontend** (`.env.local`):
```
NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1
```

### Database

The default configuration uses SQLite for development. For production deployments, configure PostgreSQL:

```
DATABASE_URL=postgresql://user:password@host:5432/benchmark
```

## Development

### Running Tests

```bash
# Backend
cd web/backend
pytest

# Frontend
cd web/frontend
npm test
```

### Code Formatting

```bash
# Backend
black web/backend
flake8 web/backend

# Frontend
npm run lint
```

## Production Deployment

### Backend

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Frontend

```bash
npm run build
npm start
```

## Citation

If you use this platform in your research, please cite:

```bibtex
@software{gemma_benchmark,
  author = {Hailey Cheng},
  title = {Gemma Benchmarking Suite: A Systematic Evaluation Framework for Large Language Models},
  year = {2025},
  url = {https://github.com/heilcheng/gemma-benchmark}
}
```

## License

This project is released under the MIT License. See the root LICENSE file for details.
