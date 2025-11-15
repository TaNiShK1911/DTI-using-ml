# DTI Prediction Platform - Documentation Index

Welcome to the DTI Prediction Platform! This index will guide you to the right documentation.

## 🚀 Getting Started (Start Here!)

**New to the project?** Follow this path:

1. **[QUICKSTART.md](QUICKSTART.md)** - Get up and running in 5 steps
2. **[INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md)** - Detailed installation instructions
3. **[CHECKLIST.md](CHECKLIST.md)** - Verify your setup step-by-step

## 📚 Core Documentation

### Overview & Architecture
- **[README.md](README.md)** - Project overview, features, and usage
- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - Complete implementation summary
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System architecture and design

### Setup & Installation
- **[INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md)** - Complete installation guide
- **[QUICKSTART.md](QUICKSTART.md)** - Quick start guide
- **[CHECKLIST.md](CHECKLIST.md)** - Setup verification checklist

### Technical Specifications
- **[requirements.txt](requirements.txt)** - Python dependencies
- **[docker-compose.yml](docker-compose.yml)** - Docker configuration
- **[.gitignore](.gitignore)** - Git ignore patterns

## 🎯 Quick Reference

### I want to...

**...install the platform**
→ [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md)

**...run the application quickly**
→ [QUICKSTART.md](QUICKSTART.md)

**...understand the architecture**
→ [ARCHITECTURE.md](ARCHITECTURE.md)

**...verify my setup**
→ [CHECKLIST.md](CHECKLIST.md)

**...see what was implemented**
→ [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)

**...troubleshoot issues**
→ [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md#troubleshooting)

**...deploy with Docker**
→ [README.md](README.md#option-2-docker)

**...understand the API**
→ [README.md](README.md#api-documentation)

**...modify the model**
→ [ARCHITECTURE.md](ARCHITECTURE.md#model-architecture-details)

**...train the model**
→ [QUICKSTART.md](QUICKSTART.md#step-3-train-the-model-optional)

## 📁 Project Structure

```
dti-prediction-platform/
│
├── 📄 Documentation (You are here!)
│   ├── INDEX.md                    # This file
│   ├── README.md                   # Main documentation
│   ├── QUICKSTART.md              # Quick start guide
│   ├── INSTALLATION_GUIDE.md      # Installation guide
│   ├── ARCHITECTURE.md            # Architecture details
│   ├── PROJECT_SUMMARY.md         # Implementation summary
│   └── CHECKLIST.md               # Setup checklist
│
├── 🔧 Setup Scripts
│   ├── setup.py                   # Setup script
│   ├── test_installation.py       # Installation test
│   ├── run_preprocessing.py       # Data preprocessing
│   ├── run_training.py            # Model training
│   └── run_backend.py             # Backend startup
│
├── 🧬 Data Processing
│   └── data/
│       ├── preprocess.py          # BindingDB preprocessing
│       └── dataset.py             # PyTorch Dataset
│
├── 🤖 Deep Learning Models
│   └── models/
│       ├── encoders.py            # Drug & protein encoders
│       ├── attention.py           # Co-attention module
│       ├── evidential.py          # Uncertainty quantification
│       └── dti_model.py           # Complete DTI model
│
├── 🎓 Training
│   └── training/
│       └── train.py               # Training pipeline
│
├── 🔌 Backend API
│   └── backend/
│       └── main.py                # FastAPI application
│
├── 🎨 Frontend
│   └── frontend/
│       ├── src/
│       │   ├── App.tsx            # Main application
│       │   ├── main.tsx           # Entry point
│       │   └── index.css          # Styles
│       ├── index.html             # HTML template
│       ├── package.json           # Dependencies
│       └── vite.config.ts         # Vite config
│
├── 🐳 Docker
│   ├── docker-compose.yml         # Orchestration
│   ├── Dockerfile.backend         # Backend container
│   └── Dockerfile.frontend        # Frontend container
│
└── 📦 Configuration
    ├── requirements.txt           # Python dependencies
    └── .gitignore                # Git ignore patterns
```

## 🎓 Learning Path

### For Beginners

1. Read [README.md](README.md) for overview
2. Follow [QUICKSTART.md](QUICKSTART.md) to get started
3. Use [CHECKLIST.md](CHECKLIST.md) to verify setup
4. Explore the application

### For Developers

1. Read [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) for implementation details
2. Study [ARCHITECTURE.md](ARCHITECTURE.md) for system design
3. Review code in `models/`, `backend/`, `frontend/`
4. Experiment with modifications

### For Researchers

1. Understand the model in [ARCHITECTURE.md](ARCHITECTURE.md)
2. Review training pipeline in `training/train.py`
3. Examine data preprocessing in `data/preprocess.py`
4. Analyze results and metrics

### For DevOps

1. Review [docker-compose.yml](docker-compose.yml)
2. Study Dockerfiles for containerization
3. Check [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md) for requirements
4. Plan deployment strategy

## 🔍 Code Reference

### Python Modules

**Data Processing:**
- `data/preprocess.py` - BindingDB data preprocessing
- `data/dataset.py` - PyTorch Dataset implementation

**Models:**
- `models/encoders.py` - Drug and protein encoders
- `models/attention.py` - Co-attention mechanism
- `models/evidential.py` - Evidential regression
- `models/dti_model.py` - Complete DTI model

**Training:**
- `training/train.py` - Training pipeline

**Backend:**
- `backend/main.py` - FastAPI application

### Frontend Files

**React Components:**
- `frontend/src/App.tsx` - Main application component
- `frontend/src/main.tsx` - Application entry point

**Styling:**
- `frontend/src/index.css` - Global styles

**Configuration:**
- `frontend/vite.config.ts` - Vite configuration
- `frontend/tsconfig.json` - TypeScript configuration

## 🛠️ Utility Scripts

### Setup & Testing
```bash
python setup.py              # Run setup
python test_installation.py  # Test installation
```

### Data & Training
```bash
python run_preprocessing.py  # Preprocess data
python run_training.py       # Train model
```

### Running the Application
```bash
python run_backend.py        # Start backend
cd frontend && npm run dev   # Start frontend
```

### Docker
```bash
docker-compose up --build    # Build and run
docker-compose down          # Stop containers
```

## 📊 Key Features

✅ **Multimodal Deep Learning**
- SMILES-based drug encoding (MolBERT)
- Protein sequence encoding (ProtBERT)
- Cross-modal co-attention

✅ **Uncertainty Quantification**
- Evidential deep learning
- Epistemic & aleatoric uncertainty
- Confidence levels

✅ **Full-Stack Application**
- React frontend with TypeScript
- FastAPI backend
- Real-time predictions

✅ **Production Ready**
- Docker deployment
- Comprehensive documentation
- Error handling & validation

## 🎯 Common Tasks

### First Time Setup
1. Install dependencies: `python setup.py`
2. Test installation: `python test_installation.py`
3. Preprocess data: `python run_preprocessing.py`
4. Start backend: `python run_backend.py`
5. Start frontend: `cd frontend && npm run dev`

### Daily Development
1. Start backend: `python run_backend.py`
2. Start frontend: `cd frontend && npm run dev`
3. Make changes
4. Test in browser: http://localhost:3000

### Training a Model
1. Ensure data is preprocessed
2. Run: `python run_training.py`
3. Wait for training to complete
4. Model saved to `checkpoints/best_model.pt`

### Deploying with Docker
1. Build: `docker-compose build`
2. Run: `docker-compose up`
3. Access: http://localhost:3000

## 🆘 Getting Help

### Troubleshooting
1. Check [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md#troubleshooting)
2. Review [CHECKLIST.md](CHECKLIST.md) for common issues
3. Run `python test_installation.py` to diagnose
4. Check error messages in terminal/console

### Understanding the Code
1. Read [ARCHITECTURE.md](ARCHITECTURE.md) for system design
2. Review [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) for implementation
3. Study code comments in source files
4. Check API docs at http://localhost:8000/docs

### Performance Issues
1. Check system requirements in [README.md](README.md)
2. Review performance tips in [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md)
3. Consider GPU acceleration
4. Reduce batch size if out of memory

## 📈 Next Steps

After getting the platform running:

1. **Explore** - Try different drug-protein pairs
2. **Experiment** - Modify hyperparameters
3. **Extend** - Add new features
4. **Deploy** - Set up production environment
5. **Research** - Analyze predictions and uncertainty

## 🌟 Key Highlights

- **Complete Implementation** - All requirements met
- **Production Ready** - Docker deployment included
- **Well Documented** - Comprehensive guides
- **Modular Design** - Easy to extend
- **Modern Stack** - Latest technologies
- **Best Practices** - Clean, tested code

## 📝 Document Versions

- **README.md** - v1.0 - Main documentation
- **QUICKSTART.md** - v1.0 - Quick start guide
- **INSTALLATION_GUIDE.md** - v1.0 - Installation guide
- **ARCHITECTURE.md** - v1.0 - Architecture details
- **PROJECT_SUMMARY.md** - v1.0 - Implementation summary
- **CHECKLIST.md** - v1.0 - Setup checklist
- **INDEX.md** - v1.0 - This document

## 🎉 You're All Set!

You now have access to all the documentation you need. Start with [QUICKSTART.md](QUICKSTART.md) and you'll be up and running in no time!

**Quick Links:**
- 🚀 [Get Started](QUICKSTART.md)
- 📖 [Full Documentation](README.md)
- 🏗️ [Architecture](ARCHITECTURE.md)
- ✅ [Checklist](CHECKLIST.md)

Happy coding! 🧬🔬💻
