# System Arechitecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            USER INTERFACE (React)                       │
│                                                                         │
│  ┌────────────────┐     ┌────────────────┐     ┌────────────────┐       │
│  │ Experiment     │     │ Progress       │     │ Results &      │       │
│  │ Setup Page     │ ──► | Monitor Page   | ──► │ Download Page  │       │
│  │                │     │                │     │                │       │
│  │ • Upload model │     │ • Technique    │     │ • Results      │       │
│  │ • Select       │     │   progress bar │     │   table        │       │
│  │   framework    │     │ • Overall      │     │ • Recommend-   │       │
│  │ • Select device│     │   progress     │     │   ation        │       │
│  │ • Select       │     │ • Live metrics │     │ • Download     │       │
│  │   dataset      │     │ • ETA display  │     │   optimized    │       │
│  │ • Set goal/    │     │                │     │   model        │       │
│  │   constraints  │     │                │     │                │       │
│  └────────────────┘     └────────────────┘     └────────────────┘       │
│         │                     ▲                      │                  │
│         │ POST /create        │ WebSocket            │ GET /download    │
│         │ POST /upload        │ progress             │                  │
│         │ POST /start         │ updates              │                  │
└─────────┼─────────────────────┼──────────────────────┼──────────────────┘
          │                     │                      │
          ▼                     │                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          BACKEND (FastAPI)                              │
│                                                                         │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ API LAYER                                                          │ │
│  │                                                                    │ │
│  │  POST   /api/experiments/create        Create new experiment       │ │
│  │  POST   /api/experiments/{id}/upload   Upload model file           │ │
│  │  GET    /api/experiments               List all experiments        │ │
│  │  GET    /api/experiments/{id}          Get experiment details      │ │
│  │  POST   /api/experiments/{id}/start    Start optimization          │ │
│  │  POST   /api/experiments/{id}/stop     Stop optimization           │ │
│  │  GET    /api/experiments/{id}/results  Get all results             │ │
│  │  GET    /api/experiments/{id}/recommendation  Get recommendation   │ │
│  │  GET    /api/experiments/{id}/download/{tech}  Download model      │ │
│  │  DELETE /api/experiments/{id}          Delete experiment           │ │
│  │  WS     /ws/experiments/{id}           WebSocket progress stream   │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                              │                                          │
│                              ▼                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ ORCHESTRATOR (Sequential Experiment Manager)                       │ │
│  │                                                                    │ │
│  │  1. Load model and dataset                                         │ │
│  │  2. Analyze model architecture                                     │ │
│  │  3. FOR EACH technique (sequential):                               │ │
│  │       a. Send progress update via WebSocket                        │ │
│  │       b. Run optimization                                          │ │
│  │       c. Evaluate metrics (accuracy, size, latency)                │ │
│  │       d. Store results in database                                 │ │
│  │       e. Save optimized model to storage                           │ │
│  │  4. Apply user constraints                                         │ │
│  │  5. Rank techniques by optimization goal                           │ │
│  │  6. Generate recommendation with reasoning                         │ │
│  │  7. Mark experiment as complete                                    │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                              │                                          │
│                              ▼                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ OPTIMIZATION ENGINE                                                │ │
│  │                                                                    │ │
│  │  ┌──────────────────────────┐  ┌──────────────────────────┐        │ │
│  │  │ PyTorch Optimizations    │  │ TensorFlow Optimizations │        │ │
│  │  ├──────────────────────────┤  ├──────────────────────────┤        │ │
│  │  │ 1. PTQ INT8              │  │ 1. PTQ INT8              │        │ │
│  │  │ 2. PTQ INT4              │  │ 2. PTQ INT4              │        │ │
│  │  │ 3. Pruning (Unstructured)│  │ 3. Pruning (Unstructured)│        │ │
│  │  │ 4. Pruning (Structured)  │  │ 4. Pruning (Structured)  │        │ │
│  │  │ 5. Knowledge Distillation│  │ 5. Knowledge Distillation│        │ │
│  │  │ 6. Quant-Aware Training  │  │ 6. Quant-Aware Training  │        │ │
│  │  │ 7. Hybrid (Prune→Quant)  │  │ 7. Hybrid (Prune→Quant)  │        │ │
│  │  └──────────────────────────┘  └──────────────────────────┘        │ │
│  │                                                                    │ │
│  │  Common Interface: BaseOptimization                                │ │
│  │    - optimize(model, config, dataset) → OptimizedResult            │ │
│  │    - requires_dataset() → bool                                     │ │
│  │    - estimate_time(model) → float                                  │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                              │                                          │
│                              ▼                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ SERVICES LAYER                                                     │ │
│  │                                                                    │ │
│  │  • ExperimentService: CRUD operations for experiments              │ │
│  │  • ModelAnalyzer: Analyze architecture, detect bottlenecks         │ │
│  │  • Recommender: Constraint validation, ranking, reasoning          │ │
│  │  • Evaluator: Accuracy, size, latency evaluation                   │ │
│  │  • PerformanceEstimator: Raspberry Pi latency/memory/power         │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                              │                                          │
│                              ▼                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ DATABASE LAYER (SQLAlchemy + PostgreSQL)                           │ │
│  │                                                                    │ │
│  │  Tables:                                                           │ │
│  │    • experiments: Experiment metadata and configuration            │ │
│  │    • optimization_runs: Individual optimization results            │ │
│  │    • model_files: Stored model file references                     │ │
│  │    • recommendations: Generated recommendations                    │ │
│  │    • experiment_progress: Real-time progress tracking              │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                              │                                          │
│                              ▼                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ FILE STORAGE (Local Filesystem)                                    │ │
│  │                                                                    │ │
│  │  dataset/preset/{dataset_name}/                                    │ │
│  │  dataset/custom/{framework}/{custom_name}/                         │ │
│  │  models/pretrained/{framework}/{model_name}.pth|.pb                │ │
│  │  models/custom/{framework}/{custom_name}.pth|.pb                   │ │
│  │  models/optimized/{experiment_id}/{technique}.pth|.pb              │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```


## Data Flow:

1. **User Creates Experiment**
   - Frontend: User fills form → POST /api/experiments/create
   - Backend: Creates experiment record in DB → Returns experiment_id

2. **User Uploads Model**
   - Frontend: User selects/uploads model → POST /api/experiments/{id}/upload
   - Backend: Validates model → Saves to storage → Updates DB

3. **User Starts Optimization**
   - Frontend: User clicks "Start" → POST /api/experiments/{id}/start
   - Backend: Orchestrator begins sequential optimization

4. **Optimization Process (Sequential)**
   - Orchestrator loads model and dataset
   - FOR EACH technique:
     - Updates progress in DB
     - Sends WebSocket message to frontend
     - Runs optimization
     - Evaluates metrics
     - Stores results in DB
     - Saves optimized model to storage

5. **Real-Time Progress Updates**
   - Backend: Orchestrator sends progress via WebSocket
   - Frontend: Receives updates → Updates UI (progress bars, metrics, ETA)

6. **Recommendations Generated**
   - Backend: After all techniques complete
     - Applies user constraints
     - Ranks techniques
     - Generates reasoning
     - Stores recommendation

7. **User Views Results**
   - Frontend: GET /api/experiments/{id}/results
   - Backend: Returns all optimization results + recommendation

8. **User Downloads Model**
   - Frontend: User selects technique → GET /api/experiments/{id}/download/{technique}
   - Backend: Serves optimized model file from storage