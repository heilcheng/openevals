# OpenEvals Web Platform

Web-based interface for the OpenEvals evaluation framework.

## Architecture

```
web/
├── backend/           # FastAPI REST API + WebSocket server
│   ├── app/
│   │   ├── api/v1/    # Versioned API endpoints
│   │   ├── models/    # SQLAlchemy ORM models
│   │   ├── schemas/   # Pydantic schemas
│   │   ├── services/  # Business logic
│   │   └── utils/     # Benchmark adapter
│   └── requirements.txt
│
└── frontend/          # Next.js 15 application
    ├── src/
    │   ├── app/       # Page routes
    │   ├── components/# React components
    │   ├── lib/       # API client
    │   └── stores/    # State management
    └── package.json
```

## Stack

**Backend:** FastAPI, SQLAlchemy, SQLite/PostgreSQL, WebSocket

**Frontend:** Next.js 15, TypeScript, Tailwind CSS, Recharts

## Setup

### Backend

```bash
cd web/backend
pip install -r requirements.txt
uvicorn app.main:app --port 8000
```

API docs: http://localhost:8000/api/docs

### Frontend

```bash
cd web/frontend
npm install
npm run dev
```

Web UI: http://localhost:3000

## API Endpoints

### Benchmarks

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /api/v1/benchmarks | List runs |
| POST | /api/v1/benchmarks | Create run |
| GET | /api/v1/benchmarks/{id} | Get details |
| POST | /api/v1/benchmarks/{id}/cancel | Cancel run |
| DELETE | /api/v1/benchmarks/{id} | Delete run |
| GET | /api/v1/benchmarks/leaderboard | Rankings |

### Models

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /api/v1/models | List configs |
| POST | /api/v1/models | Create config |
| GET | /api/v1/models/types | Available types |
| DELETE | /api/v1/models/{id} | Delete config |

### Tasks

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /api/v1/tasks | List tasks |
| GET | /api/v1/tasks/{name} | Task details |

### WebSocket

| Endpoint | Description |
|----------|-------------|
| ws://host/ws/benchmark/{run_id} | Real-time progress |

## Supported Models

| Type | Sizes |
|------|-------|
| gemma, gemma3 | 1B - 27B |
| llama3, llama3.1, llama3.2 | 1B - 405B |
| mistral, mixtral | 7B, 8x7B, 8x22B |
| qwen2, qwen2.5 | 0.5B - 72B |
| deepseek, deepseek-r1 | 1.5B - 671B |
| phi3 | Mini, Small, Medium |
| olmo | 1B, 7B |
| huggingface | Custom |

## Supported Tasks

MMLU, GSM8K, MATH, HumanEval, TruthfulQA, HellaSwag, ARC, WinoGrande, GPQA, IFEval, BBH

## Configuration

**Backend (.env):**
```
DATABASE_URL=sqlite:///./openevals.db
```

**Frontend (.env.local):**
```
NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1
```

## Production

```bash
# Backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4

# Frontend
npm run build && npm start
```

## License

MIT License
