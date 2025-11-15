# DTI Prediction Platform - Setup Checklist

Use this checklist to ensure you've completed all necessary steps.

## ✅ Pre-Installation

- [ ] Python 3.9+ installed (`python --version`)
- [ ] pip installed (`pip --version`)
- [ ] Node.js 18+ installed (`node --version`)
- [ ] npm installed (`npm --version`)
- [ ] At least 10GB free disk space
- [ ] At least 8GB RAM available
- [ ] (Optional) CUDA installed for GPU support

## ✅ Installation

- [ ] Downloaded/cloned the project
- [ ] Navigated to project directory
- [ ] Ran `python setup.py` successfully
- [ ] All Python dependencies installed
- [ ] All frontend dependencies installed
- [ ] Ran `python test_installation.py` successfully
- [ ] All tests passed

## ✅ Data Preparation

- [ ] BindingDB TSV file available
- [ ] File placed in `BindingDB_All_202511_tsv/` directory
- [ ] Ran `python run_preprocessing.py`
- [ ] Preprocessing completed without errors
- [ ] Files created in `data/processed/`:
  - [ ] `train.csv`
  - [ ] `val.csv`
  - [ ] `test.csv`

## ✅ Model Training (Optional)

- [ ] Ran `python run_training.py`
- [ ] Training completed without errors
- [ ] Model checkpoint saved to `checkpoints/best_model.pt`
- [ ] Training metrics look reasonable
- [ ] No out-of-memory errors

**Note:** You can skip training and use an untrained model for demo purposes.

## ✅ Backend Setup

- [ ] Ran `python run_backend.py`
- [ ] Backend started without errors
- [ ] Backend accessible at http://localhost:8000
- [ ] Health check works: http://localhost:8000/health
- [ ] API docs accessible: http://localhost:8000/docs
- [ ] No model loading errors (or using untrained model)

## ✅ Frontend Setup

- [ ] Opened new terminal
- [ ] Navigated to `frontend/` directory
- [ ] Ran `npm run dev`
- [ ] Frontend started without errors
- [ ] Frontend accessible at http://localhost:3000
- [ ] Page loads correctly
- [ ] No console errors in browser

## ✅ Application Testing

- [ ] Opened http://localhost:3000 in browser
- [ ] Page displays correctly
- [ ] Input form visible
- [ ] Example buttons visible
- [ ] Clicked "Aspirin + COX-2" example
- [ ] SMILES and sequence populated
- [ ] Clicked "Predict Binding Affinity"
- [ ] Loading indicator appeared
- [ ] Results displayed:
  - [ ] Affinity value shown
  - [ ] Uncertainty value shown
  - [ ] Confidence level shown
  - [ ] Attention section visible
- [ ] No errors in browser console
- [ ] No errors in backend terminal

## ✅ Functionality Testing

- [ ] Tested with custom SMILES input
- [ ] Tested with custom protein sequence
- [ ] Tested error handling (invalid SMILES)
- [ ] Tested error handling (invalid sequence)
- [ ] Tested both example buttons
- [ ] Results update correctly
- [ ] Loading states work properly

## ✅ Docker Setup (Optional)

- [ ] Docker installed
- [ ] Docker Compose installed
- [ ] Ran `docker-compose up --build`
- [ ] Both containers started successfully
- [ ] Frontend accessible at http://localhost:3000
- [ ] Backend accessible at http://localhost:8000
- [ ] Application works in Docker

## ✅ Documentation Review

- [ ] Read README.md
- [ ] Read QUICKSTART.md
- [ ] Read INSTALLATION_GUIDE.md
- [ ] Read ARCHITECTURE.md
- [ ] Read PROJECT_SUMMARY.md
- [ ] Understand the project structure
- [ ] Know where to find help

## ✅ Troubleshooting (If Needed)

- [ ] Checked error messages
- [ ] Reviewed troubleshooting sections
- [ ] Checked logs directory
- [ ] Verified all dependencies installed
- [ ] Checked port availability (3000, 8000)
- [ ] Restarted services if needed

## 🎯 Success Criteria

You've successfully set up the platform if:

✅ Backend is running on port 8000
✅ Frontend is running on port 3000
✅ You can submit predictions
✅ Results are displayed correctly
✅ No critical errors in console/terminal

## 📝 Common Issues Checklist

### Backend won't start
- [ ] Check if port 8000 is available
- [ ] Verify Python dependencies installed
- [ ] Check if model file exists (or accept untrained model)
- [ ] Review backend terminal for errors

### Frontend won't start
- [ ] Check if port 3000 is available
- [ ] Verify npm dependencies installed
- [ ] Check Node.js version
- [ ] Review frontend terminal for errors

### Predictions fail
- [ ] Backend is running
- [ ] Frontend can reach backend
- [ ] Check browser console for errors
- [ ] Verify input format (valid SMILES/sequence)
- [ ] Check backend terminal for errors

### Out of memory
- [ ] Close other applications
- [ ] Reduce batch size in training
- [ ] Use CPU instead of GPU
- [ ] Increase system swap space

### Slow performance
- [ ] Check if using GPU (if available)
- [ ] Verify model is loaded correctly
- [ ] Check system resources
- [ ] Consider using smaller inputs for testing

## 🚀 Next Steps After Setup

Once everything is working:

1. **Explore the Application**
   - [ ] Try different drug-protein pairs
   - [ ] Observe uncertainty scores
   - [ ] Compare different examples

2. **Experiment with Training**
   - [ ] Modify hyperparameters
   - [ ] Train on full dataset
   - [ ] Evaluate performance metrics

3. **Customize the Platform**
   - [ ] Add new examples
   - [ ] Modify the UI
   - [ ] Adjust model architecture
   - [ ] Add new features

4. **Deploy to Production**
   - [ ] Set up proper hosting
   - [ ] Configure domain name
   - [ ] Add authentication
   - [ ] Set up monitoring
   - [ ] Implement rate limiting

## 📊 Performance Benchmarks

Expected performance on reference hardware:

**Training (10K samples, 10 epochs):**
- GPU (RTX 3080): ~30-45 minutes
- CPU (8 cores): ~2-3 hours

**Inference:**
- GPU: ~200-300ms per prediction
- CPU: ~500-1000ms per prediction

**Memory Usage:**
- Training: ~6-8GB GPU / ~8-10GB RAM
- Inference: ~2-3GB GPU / ~4-6GB RAM

## 🎓 Learning Resources

To understand the platform better:

- [ ] Review model architecture in ARCHITECTURE.md
- [ ] Study the code in `models/` directory
- [ ] Read about evidential deep learning
- [ ] Learn about attention mechanisms
- [ ] Explore pre-trained transformers (MolBERT, ProtBERT)

## 💡 Tips for Success

1. **Start Simple**
   - Use the provided examples first
   - Test with small inputs
   - Verify each component works

2. **Monitor Resources**
   - Watch memory usage
   - Check CPU/GPU utilization
   - Monitor disk space

3. **Keep Logs**
   - Save terminal outputs
   - Check logs directory
   - Document any issues

4. **Iterate Gradually**
   - Make one change at a time
   - Test after each change
   - Keep backups of working configs

## ✨ You're Ready!

If you've checked all the boxes above, congratulations! 🎉

Your DTI Prediction Platform is fully set up and ready to use.

**Quick Start Commands:**

Terminal 1 (Backend):
```bash
python run_backend.py
```

Terminal 2 (Frontend):
```bash
cd frontend && npm run dev
```

Browser:
```
http://localhost:3000
```

Happy predicting! 🧬🔬
