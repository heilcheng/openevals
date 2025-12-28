# Gemma Benchmark Web Platform

A full interactive web platform for the Gemma Benchmark suite. Evaluate and compare language models across multiple benchmarks with real-time progress tracking, visualizations, and leaderboards.

## Features

- 🚀 **Run Benchmarks** - Configure and execute benchmarks with custom model/task selections
- 📊 **Live Progress** - Real-time WebSocket updates during benchmark execution
- 📈 **Visualizations** - Performance charts, radar plots, and heatmaps
- 🏆 **Leaderboard** - Ranked model performance across tasks
- 🎨 **Modern UI** - Beautiful dark theme with animations and effects

## Tech Stack

### Frontend
- **Next.js 15** - React framework with App Router
- **Tailwind CSS** - Utility-first CSS
- **shadcn/ui** - Reusable components
- **Framer Motion** - Animations
- **Recharts** - Data visualization
- **Zustand** - State management

### Backend
- **FastAPI** - Python async API framework
- **SQLAlchemy** - Database ORM
- **SQLite/PostgreSQL** - Database
- **WebSocket** - Real-time updates

## Quick Start

### 1. Start the Backend

```bash
cd web/backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### 2. Start the Frontend

```bash
cd web/frontend
npm install
npm run dev
```

### 3. Access the Platform

Open [http://localhost:3000](http://localhost:3000) in your browser.

## Project Structure

```
web/
├── backend/
│   ├── app/
│   │   ├── api/v1/          # API endpoints
│   │   ├── models/          # Database models
│   │   ├── schemas/         # Pydantic schemas
│   │   ├── services/        # Business logic
│   │   ├── utils/           # Utilities
│   │   └── main.py          # FastAPI app
│   └── requirements.txt
│
└── frontend/
    ├── src/
    │   ├── app/             # Next.js pages
    │   │   └── dashboard/   # Dashboard routes
    │   ├── components/      # React components
    │   │   ├── ui/          # shadcn components
    │   │   ├── charts/      # Visualization
    │   │   ├── magic/       # Magic UI effects
    │   │   └── layout/      # Layout components
    │   ├── hooks/           # Custom hooks
    │   ├── lib/             # Utilities
    │   └── stores/          # Zustand stores
    └── package.json
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/benchmarks` | List benchmark runs |
| POST | `/api/v1/benchmarks` | Create new benchmark |
| GET | `/api/v1/benchmarks/{id}` | Get benchmark details |
| POST | `/api/v1/benchmarks/{id}/cancel` | Cancel running benchmark |
| GET | `/api/v1/models` | List saved model configs |
| GET | `/api/v1/tasks` | List available tasks |
| GET | `/api/v1/benchmarks/leaderboard` | Get leaderboard |
| WS | `/ws/benchmark/{id}` | Real-time progress |

## Environment Variables

### Frontend (.env.local)
```
NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1
```

### Backend (.env)
```
DATABASE_URL=sqlite:///./gemma_benchmark.db
```

## Screenshots

The platform features a cyberpunk-inspired dark theme with:
- Animated gradient backgrounds
- Spotlight cards with hover effects
- Shimmer buttons
- Real-time progress animations
- Responsive design for all screen sizes

## Development

### Frontend Development
```bash
npm run dev    # Start development server
npm run build  # Build for production
npm run lint   # Run ESLint
```

### Backend Development
```bash
uvicorn app.main:app --reload  # Start with hot reload
```

## License

MIT License - see the root LICENSE file for details.
