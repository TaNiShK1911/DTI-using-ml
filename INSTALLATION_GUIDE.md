# Installation Guide

Complete step-by-step guide to install and run the DTI Prediction Platform.

## Prerequisites

### Required Software

1. **Python 3.9 or higher**
   - Download from: https://www.python.org/downloads/
   - Verify: `python --version`

2. **Node.js 18 or higher**
   - Download from: https://nodejs.org/
   - Verify: `node --version`

3. **pip (Python package manager)**
   - Usually comes with Python
   - Verify: `pip --version`

4. **npm (Node package manager)**
   - Comes with Node.js
   - Verify: `npm --version`

### Optional Software

- **Git** (for version control)
- **Docker** (for containerized deployment)
- **CUDA** (for GPU acceleration)

## Installation Steps

### Step 1: Clone or Download the Project

If using Git:
```bash
git clone <repository-url>
cd dti-prediction-platform
```

Or download and extract the ZIP file.

### Step 2: Run Setup Script

The setup script will check your environment and install dependencies:

```bash
python setup.py
```

This will:
- Check Python and Node.js versions
- Install Python dependencies
- Install frontend dependencies
- Create necessary directories

### Step 3: Verify Installation

Run the test script to verify everything is working:

```bash
python test_installation.py
```

This will test:
- All required packages
- Model components
- Data processing
- Backend API
- Training pipeline
- RDKit and PyTorch functionality

## Manual Installation (Alternative)

If the setup script doesn't work, follow these manual steps:

### Install Python Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- PyTorch
- Transformers (Hugging Face)
- FastAPI
- RDKit
- Pandas, NumPy, SciPy
- And other dependencies

### Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

This installs:
- React
- TypeScript
- Vite
- Axios

### Create Directories

```bash
mkdir -p data/processed
mkdir -p checkpoints
mkdir -p logs
```

## Troubleshooting

### Python Version Issues

If you have multiple Python versions:
```bash
python3.9 -m pip install -r requirements.txt
python3.9 setup.py
```

### RDKit Installation Issues

RDKit can be tricky to install. Try:

**Using conda (recommended):**
```bash
conda install -c conda-forge rdkit
```

**Using pip:**
```bash
pip install rdkit-pypi
```

### PyTorch Installation Issues

For CPU-only installation:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

For CUDA 11.8:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

For CUDA 12.1:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Transformers Model Download Issues

If you have network issues downloading models:

1. Set up a proxy:
```bash
export HTTP_PROXY=http://your-proxy:port
export HTTPS_PROXY=http://your-proxy:port
```

2. Or download models manually and place in cache:
   - Default cache: `~/.cache/huggingface/`

### Frontend Installation Issues

If npm install fails:

1. Clear npm cache:
```bash
npm cache clean --force
```

2. Delete node_modules and try again:
```bash
cd frontend
rm -rf node_modules
npm install
```

3. Try using yarn instead:
```bash
npm install -g yarn
yarn install
```

### Memory Issues

If you run out of memory during installation:

1. Close other applications
2. Increase swap space (Linux)
3. Install packages one at a time:
```bash
pip install torch
pip install transformers
pip install fastapi
# etc.
```

## Verifying GPU Support

To check if PyTorch can use your GPU:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

## Next Steps

After successful installation:

1. **Preprocess Data**
   ```bash
   python run_preprocessing.py
   ```

2. **Train Model** (optional)
   ```bash
   python run_training.py
   ```

3. **Start Backend**
   ```bash
   python run_backend.py
   ```

4. **Start Frontend** (in another terminal)
   ```bash
   cd frontend
   npm run dev
   ```

5. **Access Application**
   - Open browser to http://localhost:3000

## Docker Installation (Alternative)

If you prefer Docker:

### Prerequisites
- Docker installed
- Docker Compose installed

### Steps

1. Build and run:
```bash
docker-compose up --build
```

2. Access application:
   - Frontend: http://localhost:3000
   - Backend: http://localhost:8000

3. Stop containers:
```bash
docker-compose down
```

## System Requirements

### Minimum Requirements
- **CPU:** 4 cores
- **RAM:** 8GB
- **Storage:** 10GB free space
- **OS:** Windows 10+, macOS 10.15+, or Linux

### Recommended Requirements
- **CPU:** 8+ cores
- **RAM:** 16GB+
- **GPU:** NVIDIA GPU with 8GB+ VRAM
- **Storage:** 20GB+ free space
- **OS:** Ubuntu 20.04+ or Windows 11

## Getting Help

If you encounter issues:

1. Check the error message carefully
2. Review the troubleshooting section above
3. Run `python test_installation.py` to diagnose
4. Check the logs in the `logs/` directory
5. Consult the README.md for more information

## Common Error Messages

### "No module named 'torch'"
- Solution: `pip install torch`

### "No module named 'rdkit'"
- Solution: `conda install -c conda-forge rdkit` or `pip install rdkit-pypi`

### "CUDA out of memory"
- Solution: Reduce batch size in training config or use CPU

### "Port 8000 already in use"
- Solution: Kill the process using port 8000 or change the port

### "Cannot find module 'react'"
- Solution: `cd frontend && npm install`

## Uninstallation

To remove the installation:

1. Delete the project directory
2. Remove Python packages:
   ```bash
   pip uninstall -r requirements.txt -y
   ```
3. Remove frontend packages:
   ```bash
   cd frontend
   rm -rf node_modules
   ```

## Support

For additional support:
- Check the README.md
- Review the QUICKSTART.md
- Consult the PROJECT_SUMMARY.md
