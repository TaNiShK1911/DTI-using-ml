# System Architecture

## Overview

The DTI Prediction Platform is a full-stack application with three main layers:
1. **Frontend Layer** - React-based user interface
2. **Backend Layer** - FastAPI service for model inference
3. **Model Layer** - Deep learning components for prediction

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        USER                                  │
│                          ↓                                   │
│                    Web Browser                               │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                   FRONTEND (React)                           │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  • Input Form (SMILES + Protein Sequence)            │   │
│  │  • Results Display (Affinity + Uncertainty)          │   │
│  │  • Attention Visualization                           │   │
│  │  • Example Buttons                                   │   │
│  └──────────────────────────────────────────────────────┘   │
│                    Port: 3000                                │
└─────────────────────────────────────────────────────────────┘
                           ↓ HTTP/JSON
┌─────────────────────────────────────────────────────────────┐
│                   BACKEND (FastAPI)                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Endpoints:                                          │   │
│  │  • POST /predict - Main prediction endpoint          │   │
│  │  • GET /health - Health check                        │   │
│  │                                                       │   │
│  │  ModelService:                                       │   │
│  │  • Load trained model                                │   │
│  │  • Validate inputs                                   │   │
│  │  • Run inference                                     │   │
│  │  • Format response                                   │   │
│  └──────────────────────────────────────────────────────┘   │
│                    Port: 8000                                │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│              DEEP LEARNING MODEL (PyTorch)                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                   DTIModel                           │   │
│  │                                                       │   │
│  │  ┌─────────────┐  ┌─────────────┐                   │   │
│  │  │   Drug      │  │  Protein    │                   │   │
│  │  │  Encoder    │  │  Encoder    │                   │   │
│  │  │ (MolBERT)   │  │ (ProtBERT)  │                   │   │
│  │  └─────────────┘  └─────────────┘                   │   │
│  │         ↓                ↓                           │   │
│  │  ┌─────────────────────────────┐                    │   │
│  │  │   Co-Attention Module       │                    │   │
│  │  │  • Drug → Protein           │                    │   │
│  │  │  • Protein → Drug           │                    │   │
│  │  │  • Feature Fusion           │                    │   │
│  │  └─────────────────────────────┘                    │   │
│  │                ↓                                     │   │
│  │  ┌─────────────────────────────┐                    │   │
│  │  │ Evidential Regression Head  │                    │   │
│  │  │  • Affinity Prediction      │                    │   │
│  │  │  • Uncertainty Estimation   │                    │   │
│  │  └─────────────────────────────┘                    │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow

### 1. Training Phase

```
BindingDB TSV File
       ↓
┌──────────────────┐
│  Preprocessing   │
│  • Parse TSV     │
│  • Filter data   │
│  • p-scaling     │
│  • Validation    │
└──────────────────┘
       ↓
┌──────────────────┐
│  Data Splitting  │
│  • Train (70%)   │
│  • Val (15%)     │
│  • Test (15%)    │
└──────────────────┘
       ↓
┌──────────────────┐
│  DTIDataset      │
│  PyTorch Dataset │
└──────────────────┘
       ↓
┌──────────────────┐
│  Training Loop   │
│  • Forward pass  │
│  • Loss compute  │
│  • Backprop      │
│  • Optimization  │
└──────────────────┘
       ↓
┌──────────────────┐
│  Model Weights   │
│  best_model.pt   │
└──────────────────┘
```

### 2. Inference Phase

```
User Input (SMILES + Sequence)
       ↓
┌──────────────────┐
│  Frontend Form   │
└──────────────────┘
       ↓ HTTP POST
┌──────────────────┐
│  Backend API     │
│  /predict        │
└──────────────────┘
       ↓
┌──────────────────┐
│  Input           │
│  Validation      │
└──────────────────┘
       ↓
┌──────────────────┐
│  Drug Encoder    │
│  SMILES → Emb    │
└──────────────────┘
       ↓
┌──────────────────┐
│  Protein Encoder │
│  Seq → Emb       │
└──────────────────┘
       ↓
┌──────────────────┐
│  Co-Attention    │
│  Cross-modal     │
└──────────────────┘
       ↓
┌──────────────────┐
│  Prediction Head │
│  Affinity + Unc  │
└──────────────────┘
       ↓
┌──────────────────┐
│  JSON Response   │
│  • affinity      │
│  • uncertainty   │
│  • confidence    │
│  • attention     │
└──────────────────┘
       ↓ HTTP Response
┌──────────────────┐
│  Frontend        │
│  Results Display │
└──────────────────┘
```

## Component Details

### Frontend Components

```
App.tsx
├── State Management
│   ├── smiles (string)
│   ├── sequence (string)
│   ├── loading (boolean)
│   ├── error (string | null)
│   └── result (PredictionResult | null)
│
├── InputForm
│   ├── SMILES textarea
│   ├── Sequence textarea
│   ├── Submit button
│   └── Example buttons
│
├── ResultsView
│   ├── Affinity metric
│   ├── Uncertainty metric
│   └── Confidence badge
│
└── AttentionVisualization
    └── Attention weights display
```

### Backend Components

```
main.py
├── FastAPI App
│   ├── CORS middleware
│   └── Routes
│       ├── POST /predict
│       └── GET /health
│
└── ModelService
    ├── __init__()
    │   └── Load model
    ├── validate_smiles()
    ├── validate_sequence()
    └── predict()
        ├── Validate inputs
        ├── Run inference
        └── Format response
```

### Model Components

```
DTIModel
├── DrugEncoder
│   ├── Tokenizer
│   ├── Pre-trained transformer
│   └── Dropout
│
├── ProteinEncoder
│   ├── Tokenizer
│   ├── Pre-trained transformer
│   └── Dropout
│
├── CoAttentionModule
│   ├── Query/Key/Value projections
│   ├── Attention computation
│   └── Fusion layer
│
└── EvidentialRegressionHead
    ├── MLP layers
    └── Output layer (4 params)
```

## Technology Stack

### Frontend
- **Framework:** React 18
- **Language:** TypeScript
- **Build Tool:** Vite
- **HTTP Client:** Axios
- **Styling:** CSS

### Backend
- **Framework:** FastAPI
- **Server:** Uvicorn
- **Validation:** Pydantic
- **Language:** Python 3.9+

### Model
- **Framework:** PyTorch 2.0+
- **Transformers:** Hugging Face
- **Chemistry:** RDKit
- **Data:** Pandas, NumPy

### Deployment
- **Containerization:** Docker
- **Orchestration:** Docker Compose

## Communication Protocols

### Frontend ↔ Backend

**Request Format:**
```json
POST /predict
Content-Type: application/json

{
  "smiles": "CC(=O)Oc1ccccc1C(=O)O",
  "sequence": "MKKFFDSRR..."
}
```

**Response Format:**
```json
HTTP 200 OK
Content-Type: application/json

{
  "affinity": 7.52,
  "uncertainty": 0.34,
  "confidence_level": "High",
  "attention_weights": [[0.1, 0.2, ...], ...]
}
```

**Error Response:**
```json
HTTP 400 Bad Request
Content-Type: application/json

{
  "detail": "Invalid SMILES string"
}
```

## Model Architecture Details

### Input Processing

```
SMILES String → Tokenizer → Token IDs → Embedding → [batch, 768]
Protein Seq → Tokenizer → Token IDs → Embedding → [batch, 1024]
```

### Co-Attention

```
Drug Embedding [batch, 768]
    ↓ Linear projection
Drug Q/K/V [batch, 512]

Protein Embedding [batch, 1024]
    ↓ Linear projection
Protein Q/K/V [batch, 512]

Attention(Q, K, V) = softmax(QK^T / √d) V

Drug → Protein: Attention(Drug_Q, Protein_K, Protein_V)
Protein → Drug: Attention(Protein_Q, Drug_K, Drug_V)

Concatenate → Fusion MLP → [batch, 512]
```

### Evidential Regression

```
Fused Features [batch, 512]
    ↓ MLP
Hidden [batch, 256]
    ↓ Linear
4 Parameters [batch, 4]
    ↓ Activation
γ (mean), ν (>0), α (>0), β (>0)

Affinity = γ
Uncertainty = epistemic + aleatoric
```

## Deployment Architecture

### Local Development

```
Terminal 1: python run_backend.py → localhost:8000
Terminal 2: cd frontend && npm run dev → localhost:3000
```

### Docker Deployment

```
docker-compose up
    ↓
Backend Container (port 8000)
Frontend Container (port 3000)
    ↓
Shared volumes:
- ./checkpoints:/app/checkpoints
- ./models:/app/models
```

## Security Considerations

1. **Input Validation**
   - SMILES validation using RDKit
   - Protein sequence validation (valid amino acids)
   - Length limits on inputs

2. **Error Handling**
   - Graceful error messages
   - No sensitive information in errors
   - Proper HTTP status codes

3. **CORS**
   - Configured for frontend origin
   - Can be restricted in production

4. **Rate Limiting**
   - Not implemented in prototype
   - Should be added for production

## Performance Characteristics

### Latency
- **Encoding:** ~100-200ms per input
- **Attention:** ~50ms
- **Prediction:** ~10ms
- **Total:** ~200-300ms per request

### Throughput
- **Sequential:** ~3-5 requests/second
- **Batch:** ~10-20 requests/second (batch size 8)

### Memory
- **Model:** ~1.5GB (loaded in memory)
- **Per Request:** ~100MB (temporary)

### Scalability
- **Horizontal:** Multiple backend instances
- **Vertical:** GPU acceleration
- **Caching:** Pre-computed embeddings

## Monitoring Points

1. **Frontend**
   - API response times
   - Error rates
   - User interactions

2. **Backend**
   - Request count
   - Response times
   - Error rates
   - Model loading status

3. **Model**
   - Inference time
   - Memory usage
   - GPU utilization
   - Prediction distribution

## Future Enhancements

1. **Architecture**
   - Add caching layer (Redis)
   - Add message queue (RabbitMQ)
   - Add load balancer

2. **Model**
   - Model quantization
   - ONNX export
   - Batch prediction API

3. **Frontend**
   - Real-time visualization
   - Batch upload
   - Result history

4. **Deployment**
   - Kubernetes orchestration
   - Auto-scaling
   - Monitoring dashboard
