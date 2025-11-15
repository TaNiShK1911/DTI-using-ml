# Design Document

## Overview

The DTI Prediction Platform is a full-stack application that combines state-of-the-art deep learning with modern web technologies to predict drug-target binding affinity with uncertainty quantification and interpretability. The system architecture consists of three major layers:

1. **Data Layer**: Preprocessing pipeline for BindingDB dataset
2. **Model Layer**: Multimodal deep learning architecture with uncertainty quantification
3. **Application Layer**: Full-stack web application (React frontend + FastAPI backend)

The platform addresses key challenges in DTI prediction: multimodal data fusion, prediction uncertainty, model interpretability, and generalization to unseen molecules.

## Architecture

### High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend (React)                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Input Form   │  │ Results View │  │ Attention    │      │
│  │ (SMILES/Seq) │  │ (Affinity +  │  │ Visualization│      │
│  │              │  │  Uncertainty)│  │              │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            │ HTTP/JSON
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   Backend API (FastAPI)                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Inference Endpoint                       │   │
│  │  • Input validation                                   │   │
│  │  • Embedding generation                               │   │
│  │  • Model inference                                    │   │
│  │  • Response formatting                                │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              Deep Learning Model (PyTorch)                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   SMILES     │  │   Protein    │  │ Co-Attention │      │
│  │   Encoder    │  │   Encoder    │  │   Module     │      │
│  │  (MolBERT/   │  │  (ProtBERT/  │  │              │      │
│  │    GNN)      │  │    ESM)      │  │              │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                            │                                 │
│                            ▼                                 │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         Fusion + Prediction Head                      │   │
│  │  • Evidential regression layer                        │   │
│  │  • Affinity prediction                                │   │
│  │  • Uncertainty estimation                             │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Data Processing Pipeline

```
BindingDB TSV
     │
     ▼
┌──────────────────────────────────────────┐
│     Data Extraction & Cleaning           │
│  • Extract SMILES, Sequence, Affinity    │
│  • Filter missing values                 │
│  • Prioritize Ki > Kd > IC50             │
└──────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────┐
│     Affinity Standardization             │
│  • Convert to nanomolar                  │
│  • Apply p-scaling: pX = -log10(X)       │
└──────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────┐
│     Train/Val/Test Splitting             │
│  • Random split (70/15/15)               │
│  • Cold-drug split                       │
│  • Cold-protein split                    │
└──────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────┐
│     Preprocessed Dataset                 │
│  • Stored as PyTorch Dataset             │
│  • Cached embeddings (optional)          │
└──────────────────────────────────────────┘
```

## Components and Interfaces

### 1. Data Processing Module

**Purpose**: Extract, clean, and prepare BindingDB data for model training

**Key Classes**:
- `BindingDBParser`: Reads TSV and extracts relevant columns
- `AffinityProcessor`: Standardizes binding affinity values
- `DataSplitter`: Creates train/val/test splits including cold-start splits
- `DTIDataset`: PyTorch Dataset class for batch loading

**Interfaces**:
```python
class BindingDBParser:
    def parse(self, tsv_path: str) -> pd.DataFrame:
        """Extract SMILES, sequences, and affinity values"""
        
class AffinityProcessor:
    def unify_affinity(self, row: pd.Series) -> float:
        """Prioritize Ki > Kd > IC50 and apply p-scaling"""
        
class DataSplitter:
    def create_splits(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Create random, cold-drug, and cold-protein splits"""
        
class DTIDataset(torch.utils.data.Dataset):
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return drug SMILES, protein sequence, and affinity"""
```

### 2. Drug Encoder Module

**Purpose**: Convert SMILES strings to fixed-dimensional embeddings

**Architecture Options**:

**Option A: Transformer-based (MolBERT)**
- Use pre-trained MolBERT or ChemBERTa
- Tokenize SMILES with SMILES-specific tokenizer
- Extract [CLS] token embedding or mean pooling
- Embedding dimension: 768

**Option B: Graph Neural Network**
- Convert SMILES to molecular graph using RDKit
- Apply GNN layers (GCN, GAT, or GIN)
- Graph-level pooling for fixed representation
- Embedding dimension: 256-512

**Recommended**: Option A (MolBERT) for faster development and proven performance

**Key Classes**:
```python
class DrugEncoder(nn.Module):
    def __init__(self, model_name: str = "seyonec/PubChem10M_SMILES_BPE_450k"):
        """Initialize pre-trained SMILES transformer"""
        
    def forward(self, smiles_list: List[str]) -> torch.Tensor:
        """Encode SMILES to embeddings [batch_size, hidden_dim]"""
```

### 3. Protein Encoder Module

**Purpose**: Convert amino acid sequences to fixed-dimensional embeddings

**Architecture**:
- Use pre-trained ProtBERT or ESM-2
- Tokenize sequences with protein-specific tokenizer
- Extract [CLS] token or mean pooling over sequence
- Embedding dimension: 1024 (ProtBERT) or 1280 (ESM-2)

**Key Classes**:
```python
class ProteinEncoder(nn.Module):
    def __init__(self, model_name: str = "Rostlab/prot_bert"):
        """Initialize pre-trained protein language model"""
        
    def forward(self, sequences: List[str]) -> torch.Tensor:
        """Encode sequences to embeddings [batch_size, hidden_dim]"""
```

### 4. Co-Attention Module

**Purpose**: Learn cross-modal interactions between drug and protein representations

**Architecture**:
- Scaled dot-product attention mechanism
- Query: drug embeddings, Key/Value: protein embeddings (and vice versa)
- Bidirectional attention: drug→protein and protein→drug
- Output: attended representations + attention weights for visualization

**Mathematical Formulation**:
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V

Drug-to-Protein: A_dp = Attention(W_q^d * E_d, W_k^p * E_p, W_v^p * E_p)
Protein-to-Drug: A_pd = Attention(W_q^p * E_p, W_k^d * E_d, W_v^d * E_d)
```

**Key Classes**:
```python
class CoAttentionModule(nn.Module):
    def __init__(self, drug_dim: int, protein_dim: int, attention_dim: int):
        """Initialize attention projection layers"""
        
    def forward(self, drug_emb: torch.Tensor, protein_emb: torch.Tensor) 
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            fused_representation: Combined drug-protein features
            attention_weights: For visualization [batch, drug_tokens, protein_tokens]
            attended_drug: Drug features after attention
            attended_protein: Protein features after attention
        """
```

### 5. Evidential Regression Head

**Purpose**: Predict binding affinity with uncertainty quantification

**Architecture**:
- Input: Fused drug-protein representation
- Output: Four evidential parameters (γ, ν, α, β) representing Normal-Inverse-Gamma distribution
- Predicted affinity: γ (mean)
- Uncertainty: Derived from epistemic (α, β) and aleatoric (ν) components

**Evidential Loss Function**:
```
L = NLL + λ * KL

NLL: Negative log-likelihood of target under predicted distribution
KL: Regularization term to prevent overconfident predictions
```

**Key Classes**:
```python
class EvidentialRegressionHead(nn.Module):
    def __init__(self, input_dim: int):
        """MLP that outputs 4 evidential parameters"""
        
    def forward(self, fused_features: torch.Tensor) 
        -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            affinity_pred: Predicted pKi/pKd/pIC50 value
            uncertainty: Total uncertainty score
        """

class EvidentialLoss(nn.Module):
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, 
                evidential_params: torch.Tensor) -> torch.Tensor:
        """Compute evidential regression loss"""
```

### 6. Complete DTI Model

**Purpose**: End-to-end model integrating all components

**Key Classes**:
```python
class DTIModel(nn.Module):
    def __init__(self, config: ModelConfig):
        self.drug_encoder = DrugEncoder(config.drug_model)
        self.protein_encoder = ProteinEncoder(config.protein_model)
        self.co_attention = CoAttentionModule(...)
        self.prediction_head = EvidentialRegressionHead(...)
        
    def forward(self, smiles: List[str], sequences: List[str]) 
        -> Dict[str, torch.Tensor]:
        """
        Returns:
            affinity: Predicted binding affinity
            uncertainty: Confidence score
            attention_weights: For visualization
        """
```

### 7. Training Module

**Purpose**: Train the DTI model with evidential loss

**Key Components**:
- Optimizer: AdamW with learning rate scheduling
- Loss: Evidential regression loss
- Metrics: MSE, MAE, Pearson correlation, Spearman correlation
- Validation: Evaluate on random, cold-drug, and cold-protein splits
- Checkpointing: Save best model based on validation performance

**Key Classes**:
```python
class Trainer:
    def __init__(self, model: DTIModel, config: TrainingConfig):
        """Initialize trainer with model and hyperparameters"""
        
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch and return metrics"""
        
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate on validation set"""
        
    def train(self, train_loader: DataLoader, val_loaders: Dict[str, DataLoader]):
        """Full training loop with checkpointing"""
```

### 8. Backend API

**Purpose**: Serve model predictions via REST API

**Framework**: FastAPI (recommended for async support and automatic docs)

**Endpoints**:

```python
# API Schema
class PredictionRequest(BaseModel):
    smiles: str
    sequence: str

class PredictionResponse(BaseModel):
    affinity: float
    uncertainty: float
    confidence_level: str  # "High", "Medium", "Low"
    attention_weights: List[List[float]]
    
# Endpoints
@app.post("/predict", response_model=PredictionResponse)
async def predict_binding(request: PredictionRequest):
    """Main prediction endpoint"""
    
@app.get("/health")
async def health_check():
    """Service health check"""
```

**Key Classes**:
```python
class ModelService:
    def __init__(self, model_path: str):
        """Load trained model weights"""
        self.model = self._load_model(model_path)
        
    def predict(self, smiles: str, sequence: str) -> Dict:
        """Run inference and return results"""
        
    def _validate_inputs(self, smiles: str, sequence: str) -> bool:
        """Validate SMILES and sequence format"""
```

### 9. Frontend Application

**Purpose**: User interface for input and visualization

**Framework**: React with TypeScript

**Key Components**:

```typescript
// Main application component
interface AppState {
  smiles: string;
  sequence: string;
  prediction: PredictionResult | null;
  loading: boolean;
  error: string | null;
}

// Input form component
const InputForm: React.FC<{
  onSubmit: (smiles: string, sequence: string) => void;
}>;

// Results display component
const ResultsView: React.FC<{
  affinity: number;
  uncertainty: number;
  confidenceLevel: string;
}>;

// Attention visualization component
const AttentionHeatmap: React.FC<{
  attentionWeights: number[][];
  smiles: string;
  sequence: string;
}>;
```

**Libraries**:
- React: UI framework
- Axios: HTTP client for API calls
- D3.js or Plotly: Attention heatmap visualization
- Tailwind CSS or Material-UI: Styling

## Data Models

### Training Data Schema

```python
@dataclass
class DTIDataPoint:
    smiles: str              # Drug SMILES string
    sequence: str            # Protein amino acid sequence
    affinity: float          # Standardized pKi/pKd/pIC50
    affinity_type: str       # "Ki", "Kd", or "IC50"
    original_value: float    # Original nanomolar value
    drug_id: str            # Unique drug identifier
    protein_id: str         # Unique protein identifier
```

### Model Configuration

```python
@dataclass
class ModelConfig:
    # Encoder settings
    drug_encoder_name: str = "seyonec/PubChem10M_SMILES_BPE_450k"
    protein_encoder_name: str = "Rostlab/prot_bert"
    drug_embedding_dim: int = 768
    protein_embedding_dim: int = 1024
    
    # Co-attention settings
    attention_dim: int = 512
    num_attention_heads: int = 8
    
    # Prediction head settings
    fusion_dim: int = 512
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256])
    dropout: float = 0.3
    
    # Training settings
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 50
    evidential_coef: float = 0.1
```

## Error Handling

### Data Processing Errors

**Invalid SMILES**:
- Validate using RDKit before processing
- Log invalid entries and skip during training
- Return 400 error with message in API

**Invalid Protein Sequence**:
- Check for valid amino acid characters
- Handle non-standard amino acids by replacement or skipping
- Return 400 error with message in API

**Missing Affinity Values**:
- Filter during preprocessing
- Log statistics of filtered data

### Model Errors

**Out of Memory**:
- Implement gradient accumulation
- Reduce batch size dynamically
- Use mixed precision training (FP16)

**Convergence Issues**:
- Implement learning rate scheduling
- Add gradient clipping
- Monitor loss curves and adjust hyperparameters

**Uncertainty Calibration**:
- Validate uncertainty estimates on held-out data
- Adjust evidential coefficient if needed
- Implement temperature scaling for calibration

### API Errors

**Model Loading Failure**:
- Retry with exponential backoff
- Return 503 Service Unavailable
- Log detailed error for debugging

**Inference Timeout**:
- Set reasonable timeout limits (e.g., 30 seconds)
- Return 504 Gateway Timeout
- Implement request queuing for high load

**Invalid Input Format**:
- Validate request schema with Pydantic
- Return 400 Bad Request with specific error message
- Provide example valid inputs in error response

### Frontend Errors

**API Connection Failure**:
- Display user-friendly error message
- Implement retry mechanism
- Show loading states clearly

**Visualization Errors**:
- Handle edge cases (very long sequences, attention matrix size)
- Provide fallback text representation
- Log errors to console for debugging

## Testing Strategy

### Unit Tests

**Data Processing**:
- Test SMILES validation and parsing
- Test affinity standardization and p-scaling
- Test train/test splitting logic
- Test cold-start split creation

**Model Components**:
- Test encoder output shapes
- Test co-attention mechanism
- Test evidential loss computation
- Test uncertainty calculation

**API Endpoints**:
- Test request validation
- Test response formatting
- Test error handling

### Integration Tests

**End-to-End Model**:
- Test forward pass with sample inputs
- Test gradient flow during training
- Test model saving and loading

**API Integration**:
- Test full prediction pipeline
- Test concurrent requests
- Test error propagation

### Validation Tests

**Model Performance**:
- Evaluate on random split (baseline)
- Evaluate on cold-drug split (generalization to new drugs)
- Evaluate on cold-protein split (generalization to new proteins)
- Compare metrics: MSE, MAE, Pearson r, Spearman ρ

**Uncertainty Calibration**:
- Plot predicted uncertainty vs actual error
- Compute calibration metrics
- Verify high uncertainty for out-of-distribution inputs

**Attention Interpretability**:
- Qualitative inspection of attention maps
- Verify attention focuses on known binding sites (if available)
- Compare with domain expert knowledge

### Performance Tests

**Inference Speed**:
- Measure latency for single prediction
- Test throughput with concurrent requests
- Profile bottlenecks (encoding vs attention vs prediction)

**Memory Usage**:
- Monitor GPU memory during training
- Monitor CPU/GPU memory during inference
- Test with various input sizes

## Deployment Considerations

### Model Artifacts

- Trained model weights (PyTorch .pt or .pth file)
- Model configuration (JSON)
- Tokenizers for SMILES and protein sequences
- Preprocessing statistics (mean, std for normalization)

### Environment Setup

**Backend**:
- Python 3.8+
- PyTorch 2.0+
- Transformers library (Hugging Face)
- FastAPI + Uvicorn
- RDKit for SMILES processing

**Frontend**:
- Node.js 16+
- React 18+
- TypeScript
- Build tool (Vite or Create React App)

### Containerization

**Docker Setup**:
```dockerfile
# Backend Dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# Frontend Dockerfile
FROM node:16-alpine
WORKDIR /app
COPY package.json .
RUN npm install
COPY . .
RUN npm run build
CMD ["npm", "run", "preview"]
```

**Docker Compose**:
```yaml
version: '3.8'
services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
    environment:
      - MODEL_PATH=/app/models/dti_model.pt
      
  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    depends_on:
      - backend
    environment:
      - REACT_APP_API_URL=http://localhost:8000
```

## Technology Stack Summary

### Data Processing & Training
- **Language**: Python 3.9+
- **Deep Learning**: PyTorch 2.0+
- **Transformers**: Hugging Face Transformers
- **Chemistry**: RDKit
- **Data**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn

### Backend API
- **Framework**: FastAPI
- **Server**: Uvicorn
- **Validation**: Pydantic
- **CORS**: FastAPI middleware

### Frontend
- **Framework**: React 18 with TypeScript
- **HTTP Client**: Axios
- **Visualization**: D3.js or Plotly.js
- **Styling**: Tailwind CSS or Material-UI
- **Build Tool**: Vite

### Development Tools
- **Version Control**: Git
- **Package Management**: pip (Python), npm (JavaScript)
- **Testing**: pytest (Python), Jest (JavaScript)
- **Code Quality**: Black, isort, ESLint, Prettier
- **Containerization**: Docker, Docker Compose

## Implementation Phases

### Phase 1: Data Pipeline (Week 1)
- Parse BindingDB TSV
- Implement affinity standardization
- Create train/val/test splits
- Implement PyTorch Dataset class

### Phase 2: Model Development (Week 2-3)
- Implement drug and protein encoders
- Implement co-attention module
- Implement evidential regression head
- Integrate components into DTIModel

### Phase 3: Training & Validation (Week 3-4)
- Implement training loop with evidential loss
- Train on random split
- Evaluate on cold-start splits
- Tune hyperparameters
- Save best model

### Phase 4: Backend API (Week 4)
- Implement FastAPI endpoints
- Integrate model inference
- Add input validation and error handling
- Test API with sample requests

### Phase 5: Frontend Development (Week 5)
- Create React application structure
- Implement input form
- Implement results display
- Implement attention visualization
- Connect to backend API

### Phase 6: Integration & Testing (Week 6)
- End-to-end testing
- Performance optimization
- Bug fixes
- Documentation

### Phase 7: Deployment (Week 7)
- Containerize applications
- Set up Docker Compose
- Deploy to cloud or local server
- Final testing and validation

## Key Design Decisions & Rationale

### 1. MolBERT over GNN for Drug Encoding
**Rationale**: Pre-trained transformers provide strong baseline performance with minimal implementation complexity. GNNs can be explored as future enhancement.

### 2. FastAPI over Flask
**Rationale**: FastAPI provides automatic API documentation, async support, and built-in request validation with Pydantic, making it more suitable for production deployment.

### 3. Evidential Deep Learning for Uncertainty
**Rationale**: Evidential regression provides principled uncertainty quantification without requiring ensemble methods or multiple forward passes, making it efficient for real-time inference.

### 4. Co-Attention for Interpretability
**Rationale**: Attention mechanisms provide natural interpretability by highlighting which parts of the input contribute to predictions, addressing the black-box problem.

### 5. Cold-Start Validation
**Rationale**: Random splits can overestimate performance due to data leakage. Cold-start splits better reflect real-world scenarios where the model encounters novel drugs or proteins.

### 6. p-Scaling for Affinity Values
**Rationale**: Logarithmic transformation normalizes the wide range of affinity values and makes the distribution more suitable for regression.

### 7. Single-Chain Protein Sequences
**Rationale**: Simplifies initial implementation. Multi-chain handling can be added as future enhancement if needed.

## Future Enhancements

- **Multi-task Learning**: Predict multiple affinity types simultaneously
- **3D Structure Integration**: Incorporate protein 3D structures when available
- **Active Learning**: Suggest most informative experiments
- **Batch Prediction**: Support multiple drug-protein pairs
- **Model Explainability**: Add SHAP or integrated gradients
- **Performance Optimization**: Model quantization, ONNX export
- **User Authentication**: Add user accounts and prediction history
- **Database Integration**: Store predictions and user data
