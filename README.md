# DTI Prediction Platform

A full-stack Drug-Target Interaction (DTI) prediction platform using multimodal deep learning with uncertainty quantification.

## Features

- **Multimodal Encoding**: SMILES-based drug encoding and protein sequence encoding using pre-trained transformers
- **Co-Attention Mechanism**: Cross-modal attention for interpretable predictions
- **Uncertainty Quantification**: Evidential deep learning for confidence estimation
- **Full-Stack Application**: React frontend + FastAPI backend
- **Docker Support**: Easy deployment with Docker Compose

## Project Structure

```
.
├── data/                   # Data processing scripts
│   ├── preprocess.py      # BindingDB preprocessing
│   └── dataset.py         # PyTorch Dataset
├── models/                # Model components
│   ├── encoders.py        # Drug and protein encoders
│   ├── attention.py       # Co-attention module
│   ├── evidential.py      # Evidential regression
│   └── dti_model.py       # Complete DTI model
├── training/              # Training scripts
│   └── train.py           # Training pipeline
├── backend/               # FastAPI backend
│   └── main.py            # API endpoints
├── frontend/              # React frontend
│   └── src/
│       ├── App.tsx        # Main application
│       └── index.css      # Styles
├── requirements.txt       # Python dependencies
├── docker-compose.yml     # Docker orchestration
└── README.md             # This file
```

## Quick Start

### Prerequisites

- Python 3.9+
- Node.js 18+
- CUDA-capable GPU (recommended for training)
- Docker (optional)

### Installation

1. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

2. **Install frontend dependencies:**
```bash
cd frontend
npm install
cd ..
```

### Data Preprocessing

Process the BindingDB dataset:

```bash
python data/preprocess.py
```

This will:
- Extract SMILES, sequences, and affinity values
- Apply p-scaling transformation
- Create train/val/test splits
- Save processed data to `data/processed/`

### Training

#### 🚀 CUDA-Accelerated Training (Recommended)

**Test CUDA setup and benchmark speed:**
```bash
python test_cuda_speed.py
```

**Train with CUDA optimization (< 30 minutes):**
```bash
python run_training.py
```

**CUDA Optimizations Enabled:**
- ✅ Mixed Precision Training (FP16) - 2-3x speedup
- ✅ Batch size: 32 (effective 64 with gradient accumulation)
- ✅ Parallel data loading with 4 workers
- ✅ OneCycleLR scheduler with warmup
- ✅ Frozen pretrained encoders for faster convergence

**Expected Training Time:**
- GPU (CUDA): ~15-25 minutes for 10 epochs
- CPU: ~2-3 hours for 10 epochs
- Speedup: 6-10x with CUDA + AMP

**Configuration:**
See `config_cuda_fast.py` for tuning options:
- Adjust batch_size based on GPU memory
- Modify num_epochs for faster/slower training
- Toggle use_amp for mixed precision

The best model will be saved to `checkpoints/best_model.pt`

#### CPU Training (Fallback)

If CUDA is not available, training will automatically use CPU with reduced batch size.

### Running the Application

#### Option 1: Local Development

**Start the backend:**
```bash
python backend/main.py
```
Backend will run on http://localhost:8000

**Start the frontend:**
```bash
cd frontend
npm run dev
```
Frontend will run on http://localhost:3000

#### Option 2: Docker

```bash
docker-compose up --build
```

Access the application at http://localhost:3000

## API Documentation

### POST /predict

Predict binding affinity for a drug-protein pair.

**Request:**
```json
{
  "smiles": "CC(=O)Oc1ccccc1C(=O)O",
  "sequence": "MKKFFDSRR..."
}
```

**Response:**
```json
{
  "affinity": 7.52,
  "uncertainty": 0.34,
  "confidence_level": "High",
  "attention_weights": [[...]],
  "detailed_explanation": {
    "affinity_calculation": {
      "interpretation": "Predicted binding affinity of 7.52 on pKi/pKd/pIC50 scale",
      "binding_strength": "Strong binding (nanomolar range)"
    },
    "uncertainty_breakdown": {
      "epistemic_uncertainty": 0.12,
      "aleatoric_uncertainty": 0.22,
      "total_uncertainty": 0.34,
      "confidence_description": "Model has moderate confidence in this prediction"
    },
    "evidential_parameters": {
      "gamma": 7.52,
      "nu": 2.1,
      "alpha": 3.4,
      "beta": 0.8
    },
    "attention_analysis": {
      "attention_shape": "8 x 8",
      "max_attention": 0.95,
      "min_attention": 0.05
    },
    "input_analysis": {
      "smiles_complexity": {
        "num_atoms": 21,
        "molecular_weight": 180.16
      },
      "sequence_composition": {
        "length": 604,
        "unique_amino_acids": 20
      }
    }
  }
}
```

### GET /health

Health check endpoint.

## Model Architecture

1. **Drug Encoder**: Pre-trained SMILES transformer (MolBERT)
2. **Protein Encoder**: Pre-trained protein language model (ProtBERT)
3. **Co-Attention**: Bidirectional attention between drug and protein
4. **Evidential Head**: Uncertainty-aware regression

## Performance Metrics

The model is evaluated using:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Pearson Correlation
- Spearman Correlation
- Uncertainty Calibration

## Configuration

### Training Configuration

Edit `training/train.py` to modify:
- `batch_size`: Batch size for training
- `num_epochs`: Maximum number of epochs
- `learning_rate`: Learning rate
- `evidential_coef`: Weight for KL regularization

### Model Configuration

Edit `models/dti_model.py` to modify:
- `drug_model`: Pre-trained drug encoder
- `protein_model`: Pre-trained protein encoder
- `attention_dim`: Attention dimension
- `hidden_dims`: MLP hidden dimensions

## Troubleshooting & Debugging

### Quick Debug Mode (Recommended)

If you're having issues with model loading, use the debug scripts:

**Windows:**
```bash
# Start both frontend and backend in debug mode
start_debug.bat

# Or manually:
python debug_backend.py  # In one terminal
cd frontend && npm run dev  # In another terminal

# Test the connection:
python test_api_connection.py
```

**Linux/Mac:**
```bash
# Start backend in debug mode
python debug_backend.py

# In another terminal, start frontend
cd frontend && npm run dev

# Test the connection
python test_api_connection.py
```

See [DEBUG_GUIDE.md](DEBUG_GUIDE.md) for detailed debugging instructions.

### Common Issues

#### Backend won't start / Model loading error

**Problem:** Shape mismatch when loading checkpoint
```
RuntimeError: Error(s) in loading state_dict...
```

**Solutions:**
1. **Use debug mode** (skips model loading):
   ```bash
   python debug_backend.py
   ```

2. **Retrain the model** (creates new checkpoint):
   ```bash
   python run_training.py
   ```

3. **Delete checkpoint** (forces untrained model):
   ```bash
   del checkpoints\best_model.pt  # Windows
   rm checkpoints/best_model.pt   # Linux/Mac
   python run_backend.py
   ```

#### Frontend can't connect to backend

**Problem:** `ECONNREFUSED` errors in frontend console

**Solutions:**
1. Ensure backend is running on port 8000
2. Test backend: `python test_api_connection.py`
3. Check if port 8000 is already in use
4. Verify proxy config in `frontend/vite.config.ts`

#### Out of Memory

Reduce batch size in `training/train.py`:
```python
config = {
    'batch_size': 4,  # Reduce from 32
    ...
}
```

#### CUDA Not Available

The code automatically falls back to CPU if CUDA is not available. Training will be slower but functional.

## Examples

### Aspirin + COX-2
- SMILES: `CC(=O)Oc1ccccc1C(=O)O`
- Sequence: `MKKFFDSRREQGGSGLGSGSSGGGGSTSGLGSGYIGRVFGIGRQQVTVDEVLAEGGFAIVFLVRTSNGMK...`

### Ibuprofen + COX-1
- SMILES: `CC(C)Cc1ccc(cc1)C(C)C(=O)O`
- Sequence: `MLARALLLCAVLALSHTANPCCSHPCQNRGVCMSVGFDQYKCDCTRTGFYGENCSTPEFLTRIKLFLK...`

## License

MIT License

## Citation

If you use this platform in your research, please cite:

```
@software{dti_prediction_platform,
  title={DTI Prediction Platform},
  author={Your Name},
  year={2024}
}
```
