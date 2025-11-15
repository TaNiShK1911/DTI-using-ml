# Quick Start Guide

This guide will help you get the DTI Prediction Platform up and running quickly.

## Step 1: Install Dependencies

### Python Dependencies
```bash
pip install -r requirements.txt
```

### Frontend Dependencies
```bash
cd frontend
npm install
cd ..
```

## Step 2: Preprocess Data

Run the preprocessing script:
```bash
python run_preprocessing.py
```

This will:
- Load the BindingDB dataset
- Extract and validate SMILES and sequences
- Apply p-scaling to affinity values
- Create train/val/test splits
- Save processed data to `data/processed/`

**Note:** The preprocessing uses a subset of 10,000 samples for the prototype to speed up training.

## Step 3: Train the Model (Optional)

Train the DTI model:
```bash
python run_training.py
```

Training takes approximately 30-60 minutes on GPU (longer on CPU).

**Note:** You can skip this step and use the untrained model for demo purposes. The API will still work but predictions won't be accurate.

## Step 4: Start the Backend

In one terminal, start the backend API:
```bash
python run_backend.py
```

The API will be available at http://localhost:8000

## Step 5: Start the Frontend

In another terminal, start the frontend:
```bash
cd frontend
npm run dev
```

The frontend will be available at http://localhost:3000

## Step 6: Use the Application

1. Open http://localhost:3000 in your browser
2. Try the example inputs by clicking "Aspirin + COX-2" or "Ibuprofen + COX-1"
3. Click "Predict Binding Affinity"
4. View the results including affinity prediction, uncertainty, and confidence level

## Alternative: Docker

If you prefer Docker:

```bash
docker-compose up --build
```

Then access the application at http://localhost:3000

## Troubleshooting

### "Module not found" errors
Make sure you're in the project root directory and have installed all dependencies.

### Out of memory during training
Reduce the batch size in `training/train.py`:
```python
config = {
    'batch_size': 4,  # Reduce from 8
    ...
}
```

### Frontend can't connect to backend
Make sure the backend is running on port 8000 and check the proxy configuration in `frontend/vite.config.ts`.

### Model loading error
If you haven't trained a model yet, the API will use an untrained model for demo purposes. Train the model using `python run_training.py` for accurate predictions.

## What's Next?

- Explore the API documentation at http://localhost:8000/docs
- Try different drug-protein pairs
- Modify the model configuration in `models/dti_model.py`
- Experiment with different hyperparameters in `training/train.py`
- Add more examples to the frontend

## System Requirements

- **Minimum:** 8GB RAM, CPU
- **Recommended:** 16GB RAM, NVIDIA GPU with 8GB+ VRAM
- **Storage:** ~5GB for dependencies and data

## Expected Performance

With the prototype configuration (10,000 samples):
- **Training time:** 30-60 minutes on GPU, 2-4 hours on CPU
- **Inference time:** <1 second per prediction
- **Model size:** ~1.5GB (includes pre-trained encoders)
