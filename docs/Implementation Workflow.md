# Implementation Workflow

## Phase 1: Database & API Foundation (Days 1-2)

**Objective**: Set up database, basic API structure, and file storage

**Tasks**:
1. Initialize backend project structure
2. Set up NeonDB PostgreSQL connection
3. Create SQLAlchemy models:
   - `Experiment`
   - `OptimizationRun`
   - `ModelFile`
   - `Recommendation`
   - `ExperimentProgress`
4. Create database migration scripts
5. Implement Pydantic schemas for request/response validation
6. Set up FastAPI app with CORS middleware
7. Create basic API endpoints:
   - `POST /api/experiments/create`
   - `GET /api/experiments`
   - `GET /api/experiments/{id}`
   - `DELETE /api/experiments/{id}`
8. Create file storage directories (`dataset/`, `models/`)
9. Implement file upload endpoint: `POST /api/experiments/{id}/upload`
10. Write unit tests for API endpoints

**Validation**:
- ✅ Database schema created successfully
- ✅ Can create experiment via API
- ✅ Can upload model file
- ✅ File stored in correct directory
- ✅ Database entries created correctly
- ✅ All tests passing

---

## Phase 2: First Optimization Technique - PTQ INT8 (Days 3-4)

**Objective**: Implement complete optimization pipeline for one technique

**Tasks**:
1. Create base optimization class: `BaseOptimization` (abstract interface)
2. Implement dataset loaders:
   - Preset datasets (CIFAR-10, MNIST, etc.)
   - Custom dataset validation and loading
3. Implement PTQ INT8 for PyTorch:
   - `core/pytorch/ptq_int8.py`
   - Use `torch.quantization.quantize_dynamic` or `torch.quantization.quantize_static`
   - Evaluate accuracy before/after
   - Calculate size reduction
4. Implement PTQ INT8 for TensorFlow:
   - `core/tensorflow/ptq_int8.py`
   - Use `tf.lite.TFLiteConverter` with quantization
   - Evaluate accuracy before/after
   - Calculate size reduction
5. Create basic orchestrator: `core/orchestrator.py`
   - Load model
   - Load dataset
   - Run single optimization
   - Store results in database
6. Create API endpoint: `POST /api/experiments/{id}/start`
7. Implement result storage in `OptimizationRun` table
8. Write unit tests for PTQ INT8 (PyTorch and TensorFlow)
9. Write integration test for orchestrator

**Validation**:
- ✅ PTQ INT8 works on sample PyTorch model (e.g., ResNet18)
- ✅ PTQ INT8 works on sample TensorFlow model (e.g., MobileNet)
- ✅ Accuracy drop < 2%
- ✅ Size reduction ~4x
- ✅ Results stored in database correctly
- ✅ Optimized model saved to storage
- ✅ All tests passing

---

## Phase 3: WebSocket Real-Time Progress (Days 5-6)

**Objective**: Enable real-time progress updates to frontend

**Tasks**:
1. Implement WebSocket endpoint: `WS /ws/experiments/{id}`
2. Create WebSocket connection manager
3. Integrate progress tracking in orchestrator:
   - Send progress updates at key stages
   - Calculate ETA based on technique execution time
   - Update `ExperimentProgress` table
4. Define WebSocket message structure:
   ```python
   {
       "type": "progress_update",
       "experiment_id": "...",
       "current_technique": {
           "name": "PTQ INT8",
           "progress_percent": 65,
           "eta_seconds": 45,
           "stage": "Evaluating accuracy"
       },
       "overall_progress": {
           "completed_techniques": 2,
           "total_techniques": 7,
           "progress_percent": 28,
           "eta_seconds": 180
       },
       "metrics": {
           "original_accuracy": 0.95,
           "current_accuracy": 0.93,
           "original_size_mb": 44.7,
           "current_size_mb": 11.2
       }
   }
   ```
5. Test WebSocket connection with sample client
6. Write integration tests for WebSocket

**Validation**:
- ✅ WebSocket connection established successfully
- ✅ Progress updates received in real-time
- ✅ ETA calculations are reasonable
- ✅ Metrics updated correctly
- ✅ Connection handles disconnects gracefully
- ✅ All tests passing

---

## Phase 4: Expand Optimization Techniques (Days 7-10)

**Objective**: Implement all remaining optimization techniques

**Tasks**:

**Day 7: PTQ INT4**
1. Implement PTQ INT4 for PyTorch: `core/pytorch/ptq_int4.py`
2. Implement PTQ INT4 for TensorFlow: `core/tensorflow/ptq_int4.py`
3. Update orchestrator to run both PTQ INT8 and INT4 sequentially
4. Write unit tests
5. Validate: Size reduction ~8x, accuracy drop acceptable

**Day 8: Magnitude-based Pruning**
1. Implement unstructured pruning for PyTorch: `core/pytorch/pruning.py`
   - Use `torch.nn.utils.prune.global_unstructured`
   - 50% sparsity
2. Implement structured pruning for PyTorch
   - Use `torch.nn.utils.prune.ln_structured`
3. Implement pruning for TensorFlow: `core/tensorflow/pruning.py`
   - Use `tfmot.sparsity.keras.prune_low_magnitude`
4. Update orchestrator to run 4 techniques sequentially
5. Write unit tests
6. Validate: Sparsity achieved, size reduction, accuracy drop acceptable

**Day 9: Knowledge Distillation**
1. Implement KD for PyTorch: `core/pytorch/distillation.py`
   - Teacher: Original model
   - Student: Smaller architecture (e.g., half the channels)
   - Distillation loss: KL divergence + CE loss
2. Implement KD for TensorFlow: `core/tensorflow/distillation.py`
3. Add dataset requirement check
4. Update orchestrator to run 5 techniques
5. Write unit tests
6. Validate: Student model smaller, accuracy preserved

**Day 10: QAT and Hybrid**
1. Implement QAT for PyTorch: `core/pytorch/quantization.py`
   - Use `torch.quantization.prepare_qat`
   - Fine-tune with small learning rate
2. Implement QAT for TensorFlow: `core/tensorflow/quantization.py`
   - Use `tfmot.quantization.keras.quantize_model`
3. Implement Hybrid (Prune → Quantize):
   - `core/pytorch/hybrid.py`
   - `core/tensorflow/hybrid.py`
   - Apply pruning first, then quantization
4. Update orchestrator to run all 7 techniques
5. Write unit tests
6. Validate: All techniques complete successfully

**Validation**:
- ✅ All 7 techniques implemented and tested
- ✅ Sequential execution works correctly
- ✅ Each technique produces expected results
- ✅ All tests passing

---

## Phase 5: Recommendation Engine (Days 11-12)

**Objective**: Implement constraint validation and ranking logic

**Tasks**:
1. Create `services/recommender.py`:
   - Parse user constraints (custom or preset goal)
   - Validate each technique result against constraints
   - Rank techniques based on optimization goal:
     - Maximize Accuracy: Sort by accuracy DESC
     - Minimize Size: Sort by size ASC
     - Minimize Latency: Sort by latency ASC
     - Balanced: Calculate Pareto frontier
   - Generate reasoning for each recommendation
2. Implement constraint types:
   - Hard constraints: min_accuracy, max_size, max_latency, max_accuracy_drop
   - Preset goals: maximize_accuracy, minimize_size, minimize_latency, balanced
3. Create recommendation storage in database
4. Create API endpoint: `GET /api/experiments/{id}/recommendation`
5. Write unit tests for recommender
6. Write integration tests with different constraint scenarios

**Validation**:
- ✅ Constraints correctly filter techniques
- ✅ Ranking matches optimization goal
- ✅ Reasoning is clear and accurate
- ✅ Pareto frontier calculated correctly for "balanced" goal
- ✅ All tests passing

---

## Phase 6: Performance Estimation (Days 13-14)

**Objective**: Estimate Raspberry Pi performance without hardware

**Tasks**:
1. Create `services/performance_estimator.py`
2. Define device profiles:
   - Raspberry Pi 4: CPU specs, memory, compute throughput
   - Raspberry Pi 5: CPU specs, memory, compute throughput
   - Raspberry Pi Zero 2W: CPU specs, memory, compute throughput
3. Implement estimation algorithms:
   - **Latency**: Based on FLOPs and device compute throughput
     - `latency_ms = (model_flops / device_flops_per_sec) * 1000`
   - **Memory**: Peak memory = model_size + activation_memory
     - Estimate activation memory from layer dimensions
   - **Power**: Correlate with compute intensity
     - `power_watts = base_power + (compute_intensity * power_per_flop)`
4. Integrate estimator into orchestrator
5. Store estimates in `OptimizationRun` table
6. Write unit tests with known model architectures

**Validation**:
- ✅ Latency estimates are reasonable (within 2x of theoretical)
- ✅ Memory estimates don't exceed device limits
- ✅ Power estimates correlate with model complexity
- ✅ All tests passing

---

## Phase 7: Frontend Foundation (Days 15-17)

**Objective**: Build React frontend with upload and configuration

**Tasks**:

**Day 15: Setup & Upload Page**
1. Initialize React project with Vite + TypeScript
2. Set up Tailwind CSS
3. Create project structure (components, pages, services, hooks, types)
4. Create `ModelUpload.tsx` component:
   - File upload input (drag & drop)
   - Model source selection: Pretrained or Custom
   - If Pretrained: Dropdown (ResNet18, VGG16, VGG19, MobileNet)
   - If Custom: File upload + name input
5. Create `services/api.ts`:
   - Axios instance with base URL
   - `createExperiment()` function
   - `uploadModel()` function
6. Create `ExperimentSetupPage.tsx`:
   - Integrate ModelUpload component
   - Test upload functionality

**Day 16: Configuration Form**
1. Create `ConfigurationForm.tsx`:
   - Framework selection: PyTorch / TensorFlow (radio buttons)
   - Target device selection: Dropdown (Pi 4, Pi 5, Pi Zero 2W)
   - Dataset selection:
     - Preset: Dropdown (CIFAR-10, CIFAR-100, MNIST, Fashion-MNIST)
     - Custom: File upload
   - Optimization goal: Dropdown (Maximize Accuracy, Minimize Size, Minimize Latency, Balanced)
   - Custom constraints (optional):
     - Min Accuracy %
     - Max Size MB
     - Max Latency ms
     - Max Accuracy Drop %
   - Submit button: "Start Optimization"
2. Create TypeScript types: `types/experiment.ts`
3. Integrate form into `ExperimentSetupPage.tsx`
4. Implement form submission → API call

**Day 17: Routing & Navigation**
1. Set up React Router
2. Create pages:
   - `HomePage.tsx` (landing)
   - `ExperimentSetupPage.tsx` (upload & config)
   - `ProgressPage.tsx` (placeholder)
   - `ResultsPage.tsx` (placeholder)
3. Implement navigation flow
4. Test complete setup flow

**Validation**:
- ✅ Can upload model through UI
- ✅ Can select all configuration options
- ✅ Form validation works
- ✅ API calls successful
- ✅ Navigation between pages works

---

## Phase 8: Frontend Progress Monitoring (Days 18-20)

**Objective**: Display real-time optimization progress

**Tasks**:

**Day 18: WebSocket Connection**
1. Create `services/websocket.ts`:
   - WebSocket connection manager
   - Auto-reconnect logic
   - Message parsing
2. Create `hooks/useWebSocket.ts`:
   - Custom hook for WebSocket connection
   - State management for connection status
   - Message handlers
3. Define TypeScript types: `types/websocket.ts`

**Day 19: Progress Components**
1. Create `TechniqueProgressCard.tsx`:
   - Technique name
   - Progress bar with percentage
   - ETA display
   - Current stage text
2. Create `OverallProgressBar.tsx`:
   - Overall progress percentage
   - "Technique X of Y"
   - Overall ETA
3. Create `MetricsDisplay.tsx`:
   - Live metrics table
   - Original vs Current values
   - Accuracy, Size, Compression ratio

**Day 20: Progress Page Integration**
1. Build `ProgressPage.tsx`:
   - Connect to WebSocket
   - Display TechniqueProgressCard for current technique
   - Display OverallProgressBar
   - Display MetricsDisplay
   - Handle WebSocket messages and update UI
2. Add loading states
3. Add error handling (connection lost, optimization failed)
4. Test with backend running

**Validation**:
- ✅ WebSocket connects successfully
- ✅ Real-time progress updates display correctly
- ✅ ETA updates are accurate
- ✅ Metrics update in real-time
- ✅ UI handles disconnects gracefully
- ✅ Loading states work correctly

---

## Phase 9: Frontend Results & Download (Days 21-22)

**Objective**: Display results and enable model download

**Tasks**:

**Day 21: Results Display**
1. Create `ResultsTable.tsx`:
   - Table showing all optimization techniques
   - Columns: Technique, Accuracy, Size, Latency, Compression Ratio
   - Highlight best technique (from recommendation)
   - Color-code constraint compliance (green = pass, red = fail)
2. Create `RecommendationCard.tsx`:
   - Display recommended technique
   - Show reasoning
   - Display pros/cons
   - Show constraint compliance status
3. Create `services/api.ts` functions:
   - `getExperimentResults(id)`
   - `getRecommendation(id)`

**Day 22: Download Functionality**
1. Create `DownloadButton.tsx`:
   - Button for each technique
   - Trigger download via API
   - Show download progress
2. Implement download API call in `services/api.ts`:
   - `downloadModel(experimentId, technique)`
3. Build `ResultsPage.tsx`:
   - Fetch results on mount
   - Display ResultsTable
   - Display RecommendationCard
   - Integrate DownloadButton for each technique
4. Add loading and error states

**Validation**:
- ✅ Results display correctly
- ✅ Recommendation highlighted
- ✅ Constraint compliance shown correctly
- ✅ Download triggers file download
- ✅ Downloaded file is correct format (.pth or .pb)
- ✅ All UI states work correctly

---

## Phase 10: Integration & Testing (Days 23-25)

**Objective**: End-to-end testing and bug fixes

**Tasks**:

**Day 23: End-to-End Testing**
1. Write E2E test: `tests/e2e/test_complete_workflow.py`
   - Create experiment
   - Upload PyTorch model
   - Start optimization
   - Verify all techniques complete
   - Verify recommendation generated
   - Download optimized model
   - Verify downloaded model works
2. Write E2E test for TensorFlow workflow
3. Write E2E test with custom constraints
4. Write E2E test with custom dataset

**Day 24: Integration Testing**
1. Test all API endpoints with Postman/curl
2. Test WebSocket with multiple clients
3. Test database queries under load
4. Test file storage with large models
5. Verify error handling:
   - Invalid model format
   - Missing dataset
   - Optimization failure
   - WebSocket disconnect

**Day 25: Bug Fixes & Refinement**
1. Fix any bugs found during testing
2. Improve error messages
3. Add input validation
4. Optimize database queries
5. Add logging throughout application
6. Performance testing and optimization

**Validation**:
- ✅ Complete PyTorch workflow works E2E
- ✅ Complete TensorFlow workflow works E2E
- ✅ Custom constraints work correctly
- ✅ Error handling works for all edge cases
- ✅ Performance is acceptable
- ✅ All tests passing

---

## Phase 11: Polish & Documentation (Days 26-28)

**Objective**: Final polish and comprehensive documentation

**Tasks**:

**Day 26: UI/UX Polish**
1. Improve loading states and spinners
2. Add animations and transitions
3. Improve error messages (user-friendly)
4. Add tooltips and help text
5. Improve responsive design (mobile, tablet)
6. Accessibility improvements (ARIA labels, keyboard navigation)

**Day 27: Backend Polish**
1. Add comprehensive logging (structured logs)
2. Add request/response logging middleware
3. Improve error handling and error messages
4. Add rate limiting (optional)
5. Add API documentation (FastAPI auto-docs)
6. Code cleanup and refactoring

**Day 28: Documentation**
1. Write comprehensive README.md:
   - Project overview
   - Architecture diagram
   - Setup instructions
   - Usage guide
   - API documentation
   - Testing instructions
2. Create .env.example files
3. Write inline code documentation
4. Create user guide for frontend
5. Document optimization techniques and their use cases

**Validation**:
- ✅ UI is polished and professional
- ✅ Error messages are clear and helpful
- ✅ Logging is comprehensive
- ✅ Documentation is complete
- ✅ Setup instructions work for fresh install

---

## Phase 12: Deployment Preparation (Days 29-30)

**Objective**: Prepare for deployment and demo

**Tasks**:

**Day 29: Docker Setup**
1. Create backend Dockerfile
2. Create frontend Dockerfile
3. Create docker-compose.yml:
   - Backend service
   - Frontend service
   - PostgreSQL service (or use NeonDB)
   - Volume mounts for persistent storage
4. Test Docker setup locally
5. Write deployment documentation

**Day 30: Final Testing & Demo Prep**
1. Test complete system in Docker
2. Prepare demo models and datasets
3. Create demo script/walkthrough
4. Performance testing
5. Final bug fixes
6. Create demo video/screenshots

**Validation**:
- ✅ Docker setup works correctly
- ✅ All services start successfully
- ✅ Demo runs smoothly
- ✅ Ready for presentation

---

## TIMELINE SUMMARY

- **Phase 1-2**: Backend foundation + first optimization (Days 1-4)
- **Phase 3-4**: WebSocket + all optimizations (Days 5-10)
- **Phase 5-6**: Recommendations + performance estimation (Days 11-14)
- **Phase 7-9**: Complete frontend (Days 15-22)
- **Phase 10-12**: Testing, polish, deployment (Days 23-30)

**Total: ~30 days** for complete implementation
