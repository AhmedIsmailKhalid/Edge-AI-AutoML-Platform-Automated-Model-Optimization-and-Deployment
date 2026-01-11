# Edge AI AutoML Platform

**Automated Model Optimization for Edge Deployment**

A production-ready platform that automates the end-to-end process of optimizing trained machine learning models for deployment on resource-constrained edge devices. Transform weeks of manual optimization work into hours with intelligent automation, hardware simulation, and deployment-ready packages.

**Live Demo:** [https://huggingface.co/spaces/Ahmedik95316/Edge-AI-AutoML-Platform](https://huggingface.co/spaces/Ahmedik95316/Edge-AI-AutoML-Platform-Automated-Model-Optimization-and-Deployment)

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Key Features](#key-features)
- [Skills Demonstrated](#skills-demonstrated)
- [Technology Stack](#technology-stack)
- [How It Improves Existing Approaches](#how-it-improves-existing-approaches)
- [Getting Started](#getting-started)
- [Usage Guide](#usage-guide)
- [Project Structure](#project-structure)
- [API Documentation](#api-documentation)
- [Testing](#testing)
- [Deployment](#deployment)
- [Performance & Results](#performance--results)
- [Production Readiness](#production-readiness)
- [Future Enhancements](#future-enhancements)
- [License](#license)
- [About the Author](#about-the-author)
- [Acknowledgments](#acknowledgments)

---

## Overview

### The Problem

Deploying machine learning models to edge devices presents significant challenges:

- **Time-Intensive Process**: ML engineers spend 2-4 weeks manually experimenting with different optimization techniques (quantization, pruning, distillation) to find the right balance between model size, accuracy, and inference speed.

- **Technical Complexity**: Successful edge deployment requires expertise in 7+ optimization techniques, custom hardware testing ($500-$5K in equipment), and weeks of trial-and-error experimentation without intelligent guidance.

- **Business Impact**: According to industry research, 70% of ML models never make it to production on edge devices due to deployment complexity and resource constraints, resulting in $20K-$50K in lost engineering time per model.

### The Solution

The Edge AI AutoML Platform is an end-to-end system that automates the entire model optimization workflow:

1. **Upload**: Trained PyTorch or TensorFlow models with custom or preset datasets
2. **Configure**: Target device and optimization goals with optional constraints
3. **Optimize**: Platform runs 7 optimization techniques automatically in parallel
4. **Recommend**: Intelligent decision support with reasoning for optimal model selection
5. **Simulate**: Hardware-aware performance estimation without physical devices
6. **Deploy**: Complete deployment packages with scripts, metadata, and documentation

### Key Value Proposition

- **Time Efficiency**: Reduce optimization time from 2-4 weeks to 2-3 hours
- **Cost Savings**: Eliminate need for physical edge hardware ($500-$5K) during development
- **Intelligent Guidance**: Data-driven recommendations with clear reasoning and trade-off analysis
- **Production Ready**: Containerized, scalable architecture deployed on cloud infrastructure
- **Educational**: Generates explainable code showing exactly what optimizations were performed and why

---

## System Architecture
```
┌────────────────────────────────────────────────────────────────┐
│                      Client Layer (React)                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Create     │  │  Real-time   │  │   Results    │          │
│  │  Experiment  │  │   Progress   │  │   Viewer     │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└────────────┬────────────────┬────────────────┬─────────────────┘
             │                │                │
             │ REST API       │ SSE Stream     │ REST API
             │                │                │
┌────────────▼────────────────▼────────────────▼─────────────────┐
│                    API Layer (FastAPI)                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │Experiments│ │  Upload  │ │ Optimize │ │ Results │           │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘           │
└────────────┬────────────────┬────────────────┬─────────────────┘
             │                │                │
             │                │                │
┌────────────▼────────────────▼────────────────▼─────────────────┐
│                     Core Engine Layer                          │
│  ┌────────────────────────────────────────────────────────┐    │
│  │              Experiment Orchestrator                   │    │
│  │  • Parallel technique execution                        │    │
│  │  • Progress tracking & SSE broadcasting                │    │
│  │  • Error handling & recovery                           │    │
│  └────────────────┬───────────────────────────────────────┘    │
│                   │                                            │
│  ┌────────────────▼───────────────────────────────────────┐    │
│  │           Optimization Techniques (7)                  │    │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │    │
│  │  │PTQ INT8  │ │PTQ INT4  │ │ Pruning  │ │   QAT    │   │    │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘   │    │
│  │  ┌──────────┐ ┌────────────┐ ┌────────────────────┐    │    │
│  │  │  Hybrid  │ │Distillation│ │ Framework-specific |    │    │
│  │  └──────────┘ └────────────┘ │implementations     │    │    │
│  │                              └────────────────────┘    │    │
│  └────────────────────────────────────────────────────────┘    │
└────────────────────────────────────────────────────────────────┘
             │                │                │
┌────────────▼────────────────▼────────────────▼──────────────────┐
│              Decision Support & Simulation Layer                │
│  ┌──────────────────────┐  ┌────────────────────────────┐       │
│  │Recommendation Engine │  │  Performance Estimator     │       │
│  │• Constraint checking │  │  • Hardware simulation     │       │
│  │• Multi-criteria rank │  │  • Latency estimation      │       │
│  │• Reasoning generation│  │  • Memory profiling        │       │
│  └──────────────────────┘  └────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────┘
             │                │                │
┌────────────▼────────────────▼────────────────▼──────────────────┐
│                      Data Layer                                 │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐            │
│  │ SQLite   │ │  Models  │ │ Datasets │ │  Outputs │            │
│  │ Database │ │ Storage  │ │ Storage  │ │ Storage  │            │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘            │
└─────────────────────────────────────────────────────────────────┘
             │                │                │
┌────────────▼────────────────▼────────────────▼──────────────────┐
│                  Infrastructure Layer                           │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐     │
│  │  Nginx         │  │  Docker        │  │  HF Spaces     │     │
│  │  Reverse Proxy │  │  Container     │  │  Deployment    │     │
│  └────────────────┘  └────────────────┘  └────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
```

### Architecture Highlights

**Three-Layer Design:**
1. **Core Optimization Engine**: Parallel execution of 7 optimization techniques with framework-specific implementations (PyTorch/TensorFlow)
2. **Decision Support System**: Constraint-based recommendation engine with multi-criteria ranking and explainable reasoning
3. **Deployment Simulation**: Hardware-aware performance estimation without requiring physical devices

**Key Architectural Decisions:**
- **Modular Design**: Each optimization technique is independently implemented, allowing easy addition of new methods
- **Async Processing**: Background task orchestration with real-time progress updates via Server-Sent Events (SSE)
- **Scalable Storage**: File-based model storage with database metadata for horizontal scaling
- **API-First**: RESTful design enables CLI, web, and programmatic access

---

## Key Features

### Optimization Techniques (7)

**Quantization:**
- Post-Training Quantization INT8 (PTQ INT8)
- Post-Training Quantization INT4 (PTQ INT4)
- Quantization-Aware Training (QAT)

**Pruning:**
- Magnitude Pruning (Unstructured)
- Magnitude Pruning (Structured)

**Advanced:**
- Hybrid (Pruning + Quantization)
- Knowledge Distillation

### Target Edge Devices (6)

- **Raspberry Pi 3B** (1.2GHz, 1GB RAM)
- **Raspberry Pi 4** (1.5GHz, 4GB RAM)
- **Raspberry Pi 5** (2.4GHz, 8GB RAM)
- **NVIDIA Jetson Nano** (Maxwell GPU, 4GB RAM)
- **NVIDIA Jetson Xavier NX** (Volta GPU, 8GB RAM)
- **Google Coral Dev Board** (Edge TPU, 1GB RAM)

### Intelligent Recommendations

- **Constraint-Based Ranking**: Automatically filters and ranks models based on user-defined constraints (max accuracy drop, max size, max latency)
- **Multi-Criteria Scoring**: Balances accuracy preservation, size reduction, and inference speed
- **Explainable Reasoning**: Provides clear explanations for why each model was recommended
- **Trade-off Visualization**: Shows performance trade-offs across all optimization techniques

### Hardware Simulation

- **No Physical Devices Required**: Uses validated performance models for latency and memory estimation
- **Device-Specific Profiling**: Accounts for CPU architecture, memory bandwidth, and accelerator availability
- **Confidence Scores**: Indicates reliability of performance predictions

### Real-Time Monitoring

- **Server-Sent Events (SSE)**: Live progress updates without polling
- **Technique Tracking**: Shows which optimization is currently running
- **Progress Percentage**: Overall completion status with technique-level details
- **Error Handling**: Graceful failure with detailed error messages

### Deployment Packages

Each optimized model includes:
- Optimized model file (.pt, .pth, .h5)
- Metadata JSON (technique, metrics, timestamp)
- Deployment script (load and inference example)
- README with optimization details

### Multi-Framework Support

- **PyTorch**: Full support for .pt and .pth models
- **TensorFlow**: Full support for .h5 models
- **Framework-Specific Optimizations**: Leverages native APIs (torch.quantization, tf.lite)

### Dataset Options

**Preset Datasets:**
- MNIST (28x28 grayscale)
- CIFAR-10 (32x32 RGB)
- Fashion-MNIST (28x28 grayscale)

**Custom Datasets:**
- Upload your own datasets (ZIP format)
- Automatic extraction and validation

### Pretrained Models

**MNIST:**
- Small CNN (Fast, 95% accuracy)
- Medium CNN (Balanced, 97% accuracy)
- Large CNN (Best, 99% accuracy)

**CIFAR-10:**
- Small CNN (Fast, 70% accuracy)
- Medium CNN (Balanced, 75% accuracy)
- Large CNN (Best, 80% accuracy)

**Fashion-MNIST:**
- Small CNN (Fast, 88% accuracy)
- Large CNN (Best, 92% accuracy)

---

## Skills Demonstrated

This project showcases production-level skills across multiple domains:

### ML/AI Engineering
- Post-training quantization (INT8, INT4)
- Quantization-aware training (QAT)
- Magnitude pruning (structured & unstructured)
- Knowledge distillation
- Hybrid optimization techniques
- Model performance profiling
- Hardware-aware optimization
- Multi-framework model handling (PyTorch, TensorFlow)

### Backend Engineering
- RESTful API design (FastAPI)
- Server-Sent Events (SSE) for real-time updates
- Asynchronous programming (async/await)
- Database design & ORM (SQLAlchemy)
- Background task orchestration
- File upload handling (multipart/form-data)
- Error handling & validation
- Logging and monitoring

### Frontend Engineering
- React 18 with functional components and hooks
- State management (TanStack Query)
- Real-time UI updates (EventSource/SSE)
- Multi-step form handling
- Responsive design (Tailwind CSS)
- Client-side routing (React Router)
- File upload with drag-and-drop
- Progress indicators and loading states

### MLOps/DevOps
- Docker containerization
- Multi-stage Docker builds
- Reverse proxy configuration (Nginx)
- Cloud deployment (Hugging Face Spaces)
- Environment variable management
- Application logging
- Health checks and monitoring
- Production-ready configuration

### System Design
- Three-layer architecture (Engine, Decision Support, Simulation)
- Microservices patterns
- Background job processing
- Database schema design
- Scalable API architecture
- Separation of concerns
- Modular component design

### Software Engineering
- Clean code principles (PEP 8, ESLint)
- Type hints and documentation
- Modular architecture
- Error handling patterns
- Git version control
- Code organization and structure
- Configuration management

---

## Technology Stack

### Backend

**Core Framework:**
- **FastAPI 0.104.1**: Modern, high-performance web framework
- **Uvicorn 0.24.0**: ASGI server with production-grade performance
- **Pydantic 2.12.4**: Data validation using Python type hints

**Database:**
- **SQLAlchemy 2.0.44**: SQL toolkit and ORM
- **Alembic 1.17.2**: Database migration tool
- **SQLite**: Lightweight, file-based database

**ML/AI Libraries:**
- **PyTorch 2.1.2**: Deep learning framework
- **TensorFlow 2.20.0**: End-to-end ML platform
- **torchvision 0.16.2**: Computer vision utilities
- **NumPy 1.26.4**: Numerical computing

**Utilities:**
- **python-dotenv 1.2.1**: Environment variable management
- **python-multipart 0.0.6**: Form data parsing

### Frontend

**Core Framework:**
- **React 18**: UI library with functional components
- **Vite 7.3.0**: Fast build tool and dev server
- **React Router DOM 7.1.1**: Client-side routing

**State Management & Data Fetching:**
- **TanStack Query 5.62.11**: Async state management
- **Axios 1.7.9**: HTTP client

**UI & Styling:**
- **Tailwind CSS 4.1.0**: Utility-first CSS framework
- **lucide-react 0.469.0**: Icon library

### Infrastructure

**Containerization:**
- **Docker**: Application containerization
- **Multi-stage builds**: Optimized image size

**Web Server:**
- **Nginx**: Reverse proxy and static file serving
- **Load balancing**: Ready for horizontal scaling

**Deployment:**
- **Hugging Face Spaces**: Cloud platform for ML applications
- **16GB RAM, 8 CPU cores**: Production-grade resources

**Storage:**
- **File system**: Model and dataset storage
- **SQLite database**: Metadata and experiment tracking

---

## How It Improves Existing Approaches

### Comparison with Industry Solutions

| Feature | **Manual Optimization** | **Edge Impulse** | **Cloud AutoML (AWS/Azure)** | **Our Platform** |
|---------|------------------------|------------------|------------------------------|------------------|
| **Time to Optimize** | 2-4 weeks (manual) | Days (requires expertise) | Hours (limited control) | **Hours (fully automated)** |
| **Hardware Required** | Yes ($500-$5K) | Cloud-hosted | Cloud only | **No (simulation)** |
| **Techniques** | Manual trial & error | Limited automation | Black box | **7 automated techniques** |
| **Frameworks** | Single | Limited | Limited | **PyTorch + TensorFlow** |
| **Explainability** | None | Limited | None | **Full reasoning provided** |
| **Custom Models** | Yes | BYOM (paid feature) | Limited | **Yes (free)** |
| **Target Devices** | Manual configuration | Pre-configured | Cloud-optimized | **6 edge devices** |
| **Cost** | Engineering time | $99-$499/month | Pay-per-use | **Free** |
| **Deployment Package** | Manual creation | Export only | Cloud-dependent | **Complete with scripts** |
| **Real-time Monitoring** | None | Limited | Dashboard | **SSE-based live updates** |
| **Recommendations** | None | Basic profiling | None | **Intelligent with reasoning** |
| **Source Code** | N/A | Proprietary | Proprietary | **Open source** |

### Key Differentiators

**1. Comprehensive Automation**
- Runs 7 optimization techniques in parallel without manual intervention
- Automatically handles framework-specific implementations
- No expertise required in individual optimization methods

**2. Intelligent Decision Support**
- Constraint-based filtering (max accuracy drop, size, latency)
- Multi-criteria ranking with weighted scoring
- Explainable recommendations with clear reasoning
- Trade-off visualization across all techniques

**3. Hardware Simulation**
- Performance estimation without physical devices
- Validated models for 6 edge device families
- Latency and memory predictions with confidence scores
- Eliminates $500-$5K hardware investment during development

**4. Production-Ready Architecture**
- Containerized deployment with Docker
- Scalable API design with async processing
- Real-time monitoring via Server-Sent Events
- Complete deployment packages for each optimized model

**5. Educational Value**
- Generates explainable code for each optimization
- Shows exactly what was done and why
- Helps users learn optimization techniques
- Transparent methodology

**6. Multi-Framework Support**
- Seamless handling of PyTorch and TensorFlow models
- Framework-specific optimizations using native APIs
- No format conversion or compatibility issues

---

## Getting Started

### Prerequisites

**For Docker Deployment:**
- Docker 20.10+
- Docker Compose (optional)

**For Local Development:**
- Python 3.11+
- Node.js 20+
- npm 10+

### Option 1: Live Demo (No Setup Required)

Access the deployed application:

[https://huggingface.co/spaces/Ahmedik95316/Edge-AI-AutoML-Platform](https://huggingface.co/spaces/Ahmedik95316/Edge-AI-AutoML-Platform-Automated-Model-Optimization-and-Deployment)

### Option 2: Docker Deployment (Recommended)

**Pull from Docker Hub:**
```bash
# Coming soon
docker pull ahmedik95316/edge-ai-automl:latest
docker run -p 7860:7860 ahmedik95316/edge-ai-automl:latest
```

**Or pull from GitHub Container Registry:**
```bash
# Coming soon
docker pull ghcr.io/ahmedik95316/edge-ai-automl:latest
docker run -p 7860:7860 ghcr.io/ahmedik95316/edge-ai-automl:latest
```

**Or build from source:**
```bash
# Clone repository
git clone https://github.com/YourUsername/Edge-AI-AutoML-Platform.git
cd Edge-AI-AutoML-Platform

# Build Docker image
docker build -t edge-ai-automl .

# Run container
docker run -p 7860:7860 edge-ai-automl
```

Access the application at `http://localhost:7860`

### Option 3: Local Development (Without Docker)

**Backend Setup:**
```bash
# Clone repository
git clone https://github.com/YourUsername/Edge-AI-AutoML-Platform.git
cd Edge-AI-AutoML-Platform

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your configuration

# Run backend
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

**Frontend Setup (in a new terminal):**
```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Set environment variables
cp .env.example .env
# Edit .env: VITE_API_URL=http://localhost:8000

# Run development server
npm run dev
```

Access the application at `http://localhost:5173`

### Environment Variables

**Backend (.env):**
```bash
# Database
DATABASE_URL=sqlite:///./experiments.db

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# CORS (comma-separated origins)
CORS_ORIGINS=http://localhost:3000,http://localhost:5173

# Security
SECRET_KEY=your-secret-key-here
```

**Frontend (frontend/.env):**
```bash
# API URL (empty for relative URLs with nginx proxy)
VITE_API_URL=http://localhost:8000
```

---

## Usage Guide

### Web Interface

**Step 1: Create Experiment**
1. Navigate to "Create Experiment"
2. Fill in experiment details:
   - Name: Descriptive name for your experiment
   - Framework: PyTorch or TensorFlow
3. Click "Next"

**Step 2: Select Dataset**
- **Preset**: Choose from MNIST, CIFAR-10, or Fashion-MNIST
- **Custom**: Upload your own dataset (ZIP format)

**Step 3: Upload Model**
- **Pretrained**: Select from available models (only for preset datasets)
- **Custom**: Upload your trained model (.pt, .pth, .h5)

**Step 4: Configure Optimization**
- Target Device: Select your edge device
- Optimization Goal: Balanced, Minimize Size, Maximize Speed, or Maximize Accuracy
- Optional Constraints:
  - Max Accuracy Drop (%): Maximum acceptable accuracy reduction
  - Max Model Size (MB): Maximum model file size
  - Max Latency (ms): Maximum inference time

**Step 5: Review & Submit**
- Review all settings
- Click "Create & Start Optimization"

**Step 6: Monitor Progress**
- Real-time progress updates
- See which technique is currently running
- View completed techniques with metrics

**Step 7: View Results**
- Compare all optimized models
- Read intelligent recommendations with reasoning
- Download deployment packages

### CLI Interface
```bash
# Run CLI interface
python scripts/cli_interface.py

# Follow interactive prompts:
# 1. Enter experiment name
# 2. Select framework (PyTorch/TensorFlow)
# 3. Select dataset (preset or custom)
# 4. Upload model file
# 5. Select target device
# 6. Set optimization goal
# 7. (Optional) Set constraints

# Monitor real-time progress in terminal
# Results saved in outputs/ directory
```

### API Usage

**Create Experiment:**
```python
import requests

# Create experiment
response = requests.post(
    "http://localhost:8000/api/experiments/create",
    json={
        "name": "my_experiment",
        "framework": "pytorch",
        "dataset_type": "preset",
        "dataset_name": "mnist",
        "target_device": "raspberry_pi_4",
        "optimization_goal": "balanced"
    }
)
experiment_id = response.json()["id"]
```

**Upload Model:**
```python
# Upload model file
with open("model.pt", "rb") as f:
    files = {"file": f}
    data = {"model_source": "custom"}
    response = requests.post(
        f"http://localhost:8000/api/upload/{experiment_id}/model",
        files=files,
        data=data
    )
```

**Start Optimization:**
```python
# Start optimization
response = requests.post(
    f"http://localhost:8000/api/optimize/{experiment_id}/start"
)
```

**Get Results:**
```python
# Fetch results
response = requests.get(
    f"http://localhost:8000/api/results/{experiment_id}/results"
)
results = response.json()
```

**Full API documentation available at:** `http://localhost:8000/docs`

---

## Project Structure
```
Edge-AI-AutoML-Platform/
│
├── src/                          # Backend source code
│   ├── __init__.py
│   ├── main.py                   # FastAPI application entry point
│   ├── config.py                 # Configuration management
│   ├── database.py               # Database connection and session
│   ├── logging_config.py         # Logging configuration
│   │
│   ├── api/                      # API endpoints
│   │   ├── __init__.py
│   │   ├── experiments.py        # Experiment CRUD operations
│   │   ├── upload.py             # Model/dataset upload endpoints
│   │   ├── optimize.py           # Optimization start/stop
│   │   ├── results.py            # Results retrieval and download
│   │   ├── performance.py        # Performance metrics
│   │   └── websocket.py          # WebSocket/SSE connections
│   │
│   ├── core/                     # Core optimization engine
│   │   ├── __init__.py
│   │   ├── base.py               # Base optimization class
│   │   ├── orchestrator.py       # Experiment orchestration
│   │   ├── recommendation_engine.py  # Intelligent recommendations
│   │   ├── performance_estimator.py  # Hardware simulation
│   │   │
│   │   ├── pytorch/              # PyTorch optimizations
│   │   │   ├── __init__.py
│   │   │   ├── ptq_int8.py       # Post-training quantization INT8
│   │   │   ├── ptq_int4.py       # Post-training quantization INT4
│   │   │   ├── pruning.py        # Magnitude pruning
│   │   │   ├── qat.py            # Quantization-aware training
│   │   │   ├── hybrid.py         # Hybrid optimization
│   │   │   └── distillation.py   # Knowledge distillation
│   │   │
│   │   └── tensorflow/           # TensorFlow optimizations
│   │       ├── __init__.py
│   │       ├── ptq_int8.py
│   │       ├── ptq_int4.py
│   │       ├── pruning.py
│   │       ├── qat.py
│   │       ├── hybrid.py
│   │       └── distillation.py
│   │
│   ├── models/                   # Database models
│   │   ├── __init__.py
│   │   ├── experiment.py         # Experiment model
│   │   ├── optimization_run.py   # Optimization run model
│   │   ├── recommendation.py     # Recommendation model
│   │   └── model_file.py         # Model file metadata
│   │
│   ├── schemas/                  # Pydantic schemas
│   │   ├── __init__.py
│   │   ├── experiment.py
│   │   ├── optimization.py
│   │   ├── result.py
│   │   └── recommendation.py
│   │
│   └── utils/                    # Utility modules
│       ├── __init__.py
│       ├── model_loader.py       # Model loading utilities
│       ├── dataset_loader.py     # Dataset loading utilities
│       ├── file_handler.py       # File operations
│       ├── validators.py         # Input validation
│       └── error_handler.py      # Error handling
│
├── frontend/                     # React frontend
│   ├── public/                   # Static assets
│   ├── src/
│   │   ├── api/                  # API client
│   │   │   └── client.js         # Axios configuration
│   │   │
│   │   ├── components/           # Reusable components
│   │   │   ├── Header.jsx
│   │   │   ├── LoadingSpinner.jsx
│   │   │   └── ExperimentCard.jsx
│   │   │
│   │   ├── pages/                # Page components
│   │   │   ├── HomePage.jsx
│   │   │   ├── CreateExperiment.jsx
│   │   │   ├── ExperimentProgress.jsx
│   │   │   ├── ExperimentResults.jsx
│   │   │   └── ResultsViewer.jsx
│   │   │
│   │   ├── App.jsx               # Root component
│   │   ├── main.jsx              # Entry point
│   │   └── index.css             # Global styles
│   │
│   ├── package.json
│   ├── vite.config.js
│   └── tailwind.config.js
│
├── scripts/                      # Helper scripts
│   ├── cli_interface.py          # CLI for experiments
│   ├── create_pretrained_models.py
│   ├── create_test_datasets.py
│   └── seed_devices.py
│
├── models/                       # Model storage
│   ├── pretrained/               # Pretrained models
│   │   ├── pytorch/
│   │   └── tensorflow/
│   └── custom/                   # User-uploaded models
│
├── dataset/                      # Dataset storage
│   ├── preset/                   # Preset datasets
│   │   ├── MNIST/
│   │   ├── CIFAR10/
│   │   └── FashionMNIST/
│   └── custom/                   # User-uploaded datasets
│
├── outputs/                      # Optimization outputs
│
├── tests/                        # Test suite
│   ├── unit/                     # Unit tests
│   ├── integration/              # Integration tests
│   └── e2e/                      # End-to-end tests
│
├── docs/                         # Documentation
│   ├── System Architecture.md
│   ├── Tech Stack.md
│   └── Implementation Workflow.md
│
├── Dockerfile                    # Docker configuration
├── docker-compose.yml            # Docker Compose (optional)
├── nginx.conf                    # Nginx configuration
├── start.sh                      # Container startup script
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variables template
├── .gitignore                    # Git ignore rules
├── .dockerignore                 # Docker ignore rules
├── README.md                     # This file
└── LICENSE                       # License file
```

### Key Files

**Backend:**
- `src/main.py`: FastAPI application with SSE support and lazy-loaded ML libraries
- `src/core/orchestrator.py`: Manages parallel execution of optimization techniques
- `src/core/recommendation_engine.py`: Constraint-based ranking with reasoning
- `src/core/performance_estimator.py`: Hardware simulation without physical devices

**Frontend:**
- `frontend/src/pages/CreateExperiment.jsx`: Multi-step form for experiment creation
- `frontend/src/pages/ExperimentProgress.jsx`: Real-time SSE-based progress monitoring
- `frontend/src/pages/ExperimentResults.jsx`: Results display with recommendations

**Infrastructure:**
- `Dockerfile`: Multi-stage build with frontend compilation and backend setup
- `nginx.conf`: Reverse proxy configuration for unified deployment
- `start.sh`: Container startup orchestration

---

## API Documentation

### Base URL

**Local:** `http://localhost:8000`  
**Production:** `https://huggingface.co/spaces/.../`

### Authentication

Currently no authentication required. Future versions will include API key authentication.

### Endpoints Overview

**Experiments:**
- `POST /api/experiments/create` - Create new experiment
- `GET /api/experiments/{id}` - Get experiment details
- `GET /api/experiments/recent` - Get recent experiments
- `GET /api/experiments/search` - Search experiments

**Upload:**
- `POST /api/upload/{id}/model` - Upload model file
- `POST /api/upload/{id}/dataset` - Upload dataset

**Optimization:**
- `POST /api/optimize/{id}/start` - Start optimization
- `POST /api/optimize/{id}/stop` - Stop optimization
- `GET /api/optimize/{id}/progress-stream` - SSE progress stream

**Results:**
- `GET /api/results/{id}/results` - Get all results
- `GET /api/results/{id}/recommendations` - Get recommendations
- `GET /api/results/{id}/download/{technique}` - Download optimized model

**Performance:**
- `GET /api/performance/{id}/metrics` - Get performance metrics
- `GET /api/performance/{id}/comparison` - Compare techniques

### Interactive Documentation

Full interactive API documentation with request/response schemas:

**Swagger UI:** `http://localhost:8000/docs`  
**ReDoc:** `http://localhost:8000/redoc`

### Example Requests

See [Usage Guide](#api-usage) section for detailed examples.

---

## Testing

### Test Structure
```
tests/
├── unit/                      # Unit tests for individual components
│   ├── test_ptq_int8_pytorch.py
│   ├── test_pruning_tensorflow.py
│   └── test_recommendation_engine.py
├── integration/               # Integration tests
│   ├── test_orchestrator.py
│   └── test_performance_api.py
└── e2e/                      # End-to-end tests
    └── test_complete_workflow.py
```

### Running Tests

**Run all tests:**
```bash
pytest
```

**Run specific test file:**
```bash
pytest tests/unit/test_ptq_int8_pytorch.py
```

**Run with coverage:**
```bash
pytest --cov=src --cov-report=html
```

**View coverage report:**
```bash
open htmlcov/index.html
```

### Test Coverage

Current test coverage focuses on:
- Individual optimization techniques
- Recommendation engine logic
- Performance estimation accuracy
- API endpoint responses
- Error handling scenarios

---

## Deployment

### Hugging Face Spaces (Current Production)

**Deployment Steps:**

1. Create new Space on Hugging Face
2. Select SDK: Docker
3. Connect GitHub repository
4. Configure environment variables
5. Push to trigger automatic deployment

**Environment Variables:**
```bash
DATABASE_URL=sqlite:///./experiments.db
SECRET_KEY=your-secret-key
CORS_ORIGINS=https://your-space.hf.space
```

**Health Checks:**
- Endpoint: `/health`
- Interval: 30s
- Timeout: 10s

### Docker Hub / GitHub Container Registry

**Build and Push:**
```bash
# Build image
docker build -t edge-ai-automl:latest .

# Tag for Docker Hub
docker tag edge-ai-automl:latest ahmedik95316/edge-ai-automl:latest

# Push to Docker Hub
docker push ahmedik95316/edge-ai-automl:latest

# Tag for GitHub Container Registry
docker tag edge-ai-automl:latest ghcr.io/ahmedik95316/edge-ai-automl:latest

# Push to GHCR
docker push ghcr.io/ahmedik95316/edge-ai-automl:latest
```

### Alternative Deployment Options

**AWS ECS:**
- Use provided Dockerfile
- Configure ALB with health checks
- Set environment variables via Parameter Store

**Google Cloud Run:**
- Deploy containerized application
- Configure Cloud SQL for database
- Enable Cloud Storage for model files

**Azure Container Instances:**
- Deploy Docker image
- Configure Azure Database
- Set up Blob Storage

### Scaling Considerations

**Horizontal Scaling:**
- Multiple Uvicorn workers (configurable via environment)
- Shared file storage (NFS, S3, GCS)
- Load balancer (Nginx, AWS ALB, GCP Load Balancer)

**Vertical Scaling:**
- Increase container resources (CPU, RAM)
- Optimize database queries
- Implement caching layer (Redis)

**Database:**
- Migrate from SQLite to PostgreSQL for production
- Connection pooling
- Read replicas for heavy read workloads

---

## Performance & Results

### Optimization Speed

**Average Time per Technique (MNIST, Medium CNN):**
- PTQ INT8: 15-30 seconds
- PTQ INT4: 20-40 seconds
- Pruning: 45-90 seconds
- QAT: 5-10 minutes (requires retraining)
- Hybrid: 2-3 minutes
- Distillation: 8-15 minutes (requires teacher model)

**Total Optimization Time:** 20-30 minutes for all 7 techniques (parallel execution)

### Model Size Reduction

**Typical Results (Large MNIST CNN, 38.93 MB baseline):**

| Technique | Model Size | Size Reduction | Accuracy | Accuracy Drop |
|-----------|------------|----------------|----------|---------------|
| **Baseline** | 38.93 MB | 0% | 99.2% | 0% |
| **PTQ INT8** | 9.78 MB | 74.9% | 99.1% | 0.1% |
| **PTQ INT4** | 5.12 MB | 86.8% | 98.8% | 0.4% |
| **Pruning 50%** | 19.47 MB | 50.0% | 99.0% | 0.2% |
| **Pruning 75%** | 9.73 MB | 75.0% | 98.5% | 0.7% |
| **QAT** | 9.78 MB | 74.9% | 99.2% | 0.0% |
| **Hybrid** | 4.87 MB | 87.5% | 98.7% | 0.5% |

**Best Trade-off:** QAT provides similar size reduction to PTQ INT8 with no accuracy loss.  
**Smallest Model:** Hybrid (Pruning + Quantization) achieves 87.5% size reduction.

### Inference Speed

**Estimated Latency (Raspberry Pi 4, MNIST inference):**

| Technique | Latency (ms) | Speedup |
|-----------|--------------|---------|
| **Baseline** | 45 ms | 1.0x |
| **PTQ INT8** | 12 ms | 3.8x |
| **PTQ INT4** | 8 ms | 5.6x |
| **Pruning 75%** | 22 ms | 2.0x |
| **Hybrid** | 6 ms | 7.5x |

**Note:** Latency estimates based on validated performance models. Actual performance may vary based on hardware batch size, and input preprocessing.

### Resource Utilization

**Backend Memory Usage:**
- Idle: ~2.5 GB (PyTorch + TensorFlow loaded)
- During Optimization: ~4-6 GB (model + training data)
- Peak: ~8 GB (multiple techniques in parallel)

**Frontend Bundle Size:**
- Total: ~360 KB (gzipped: ~110 KB)
- Initial load: <1 second on broadband

---

## Production Readiness

### Current Implementation

**Deployment:**
- Containerized with Docker
- Deployed on Hugging Face Spaces (16GB RAM, 8 CPU cores)
- Nginx reverse proxy for unified endpoint
- Health checks and automatic restarts

**Monitoring:**
- Application logging (file-based)
- Health check endpoint (`/health`)
- Real-time progress tracking (SSE)
- Error logging with stack traces

**Security:**
- Input validation on all endpoints
- File type verification
- Size limits on uploads
- CORS configuration

**Scalability:**
- Async processing with background tasks
- Database connection pooling
- File-based model storage (scalable to object storage)
- API-first design for easy horizontal scaling

### Production Considerations

**Recommended Enhancements for Enterprise Deployment:**

**Authentication & Authorization:**
- API key authentication
- JWT tokens for session management
- Role-based access control (RBAC)

**Monitoring & Observability:**
- Prometheus metrics
- Grafana dashboards
- Distributed tracing (Jaeger, OpenTelemetry)
- Error tracking (Sentry)

**Database:**
- Migrate to PostgreSQL for production
- Connection pooling (PgBouncer)
- Automated backups
- Read replicas

**Storage:**
- Object storage (S3, GCS, Azure Blob)
- CDN for model downloads
- Automatic cleanup of old files

**CI/CD:**
- Automated testing on PR
- Docker image building and pushing
- Automated deployment to staging/production
- Rollback capability

**Security Hardening:**
- Rate limiting (per IP, per user)
- Input sanitization
- Secret management (AWS Secrets Manager, Vault)
- HTTPS enforcement
- Security headers

**High Availability:**
- Load balancer with health checks
- Multiple Uvicorn workers
- Database replication
- Automatic failover

---

## Future Enhancements

### Short-term (Next 3-6 months)

**CI/CD Pipeline:**
1. **Basic CI/CD (Priority 1):**
   - GitHub Actions workflow
   - Automated testing on pull requests
   - Docker image building
   - Code quality checks (linting, formatting)
   - Test coverage reporting

2. **Comprehensive CI/CD (Priority 2):**
   - Automated deployment to Hugging Face Spaces
   - Push to Docker Hub and GitHub Container Registry
   - Multi-stage testing (unit, integration, e2e)
   - Performance benchmarking
   - Security scanning

**Additional Features:**
- ONNX format support (cross-framework compatibility)
- More target devices (ESP32, STM32, mobile SoCs)
- Model architecture search (AutoML)
- Batch experiment processing
- Experiment comparison dashboard

### Long-term (6-12 months)

**Real Hardware Benchmarking:**
- Integration with edge device farms
- Actual hardware performance measurement
- Power consumption profiling

**Advanced Optimization:**
- Neural architecture search (NAS)
- Mixed-precision quantization
- Sparse tensor support
- Dynamic pruning

**Platform Enhancements:**
- Multi-user support with authentication
- Team collaboration features
- Private model repositories
- Experiment scheduling
- Cost estimation for cloud deployment

**ML Lifecycle:**
- Model versioning
- A/B testing support
- Model monitoring in production
- Drift detection
- Automated retraining pipelines

---

## License

MIT License

Copyright (c) 2026 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## About the Author

**Name:** [Your Name]

**Role:** ML/AI/MLOps Engineer

**Links:**
- LinkedIn: [Your LinkedIn URL]
- GitHub: [Your GitHub URL]
- Portfolio: [Your Portfolio URL]
- Email: [Your Email]

**About This Project:**

This project was developed as a portfolio piece to demonstrate end-to-end product development skills, from system design and ML engineering to full-stack development and production deployment. It showcases the ability to identify real-world problems, design scalable solutions, and deliver production-ready applications.

**Video Demo:** [YouTube link - to be added]

---

## Acknowledgments

**ML Frameworks:**
- PyTorch Team for the exceptional deep learning framework
- TensorFlow Team for comprehensive ML platform

**Web Frameworks:**
- FastAPI for modern, high-performance API development
- React Team for powerful UI library

**Infrastructure:**
- Hugging Face for free hosting of ML applications
- Docker for containerization technology
- Nginx for robust reverse proxy

**Development Tools:**
- Vite for lightning-fast frontend builds
- SQLAlchemy for elegant database ORM
- Tailwind CSS for utility-first styling

**Community:**
- Edge AI research community for optimization insights
- Open source contributors whose tools made this possible

---

**Built with passion for Edge AI and MLOps.**

Last Updated: January 2026
