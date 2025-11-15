# Implementation Plan

This plan breaks down the DTI Prediction Platform into discrete, actionable coding tasks. Each task builds incrementally on previous work, ensuring continuous integration of components.

## Task List

- [ ] 1. Set up project structure and dependencies
  - Create directory structure: `data/`, `models/`, `training/`, `backend/`, `frontend/`
  - Create Python virtual environment and requirements.txt with core dependencies (PyTorch, Transformers, FastAPI, RDKit, pandas)
  - Create package.json for frontend with React, TypeScript, Axios, and visualization libraries
  - Initialize Git repository with .gitignore for Python and Node.js
  - _Requirements: 11.1, 11.2_

- [ ] 2. Implement BindingDB data preprocessing pipeline
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

- [ ] 2.1 Create BindingDB parser module
  - Write `BindingDBParser` class to read TSV file and extract Ligand SMILES, BindingDB Target Chain Sequence 1, Ki (nM), Kd (nM), and IC50 (nM) columns
  - Implement filtering logic to remove rows with missing SMILES or sequence
  - Add validation for SMILES strings using RDKit
  - _Requirements: 6.1, 6.2, 6.4_

- [ ] 2.2 Implement affinity standardization module
  - Write `AffinityProcessor` class with `unify_affinity()` method that prioritizes Ki > Kd > IC50
  - Implement p-scaling transformation: pX = -log10(X) for nanomolar values
  - Add unit conversion if values are in different units
  - Store original affinity type and value for reference
  - _Requirements: 6.3, 6.5, 4.2, 4.3_

- [ ] 2.3 Create data splitting module
  - Write `DataSplitter` class to create three split types: random (70/15/15), cold-drug, and cold-protein
  - Implement cold-drug split: identify unique drugs, split drugs into train/test, assign all interactions accordingly
  - Implement cold-protein split: identify unique proteins, split proteins into train/test, assign all interactions accordingly
  - Save split indices or identifiers to ensure reproducibility
  - _Requirements: 7.1, 7.2, 7.4_

- [ ] 2.4 Implement PyTorch Dataset class
  - Write `DTIDataset` class extending `torch.utils.data.Dataset`
  - Implement `__getitem__()` to return dictionary with SMILES, sequence, affinity, and metadata
  - Implement `__len__()` method
  - Add optional caching mechanism for preprocessed data
  - _Requirements: 6.1, 6.2, 6.5_

- [ ]* 2.5 Write data preprocessing tests
  - Test SMILES validation and filtering
  - Test affinity prioritization logic with sample data
  - Test p-scaling calculation accuracy
  - Test split creation ensures no overlap in cold-start conditions
  - _Requirements: 6.3, 6.4, 7.4_

- [ ] 3. Implement drug encoder module
  - _Requirements: 1.1, 1.2, 1.3, 1.4_

- [ ] 3.1 Create SMILES encoder with pre-trained transformer
  - Write `DrugEncoder` class extending `nn.Module`
  - Load pre-trained MolBERT or ChemBERTa model from Hugging Face
  - Implement `forward()` method that tokenizes SMILES and extracts [CLS] token embedding
  - Add batch processing support
  - Handle variable-length SMILES with padding
  - _Requirements: 1.1, 1.2, 1.3_

- [ ] 3.2 Add drug encoder configuration and freezing options
  - Implement option to freeze pre-trained weights or allow fine-tuning
  - Add dropout layer after encoder for regularization
  - Create configuration dataclass for encoder hyperparameters
  - _Requirements: 1.1, 1.5_

- [ ]* 3.3 Write drug encoder tests
  - Test output shape is [batch_size, embedding_dim]
  - Test handling of invalid SMILES
  - Test batch processing with variable-length inputs
  - _Requirements: 1.3_

- [ ] 4. Implement protein encoder module
  - _Requirements: 2.1, 2.2, 2.3, 2.4_

- [ ] 4.1 Create protein sequence encoder with pre-trained PLM
  - Write `ProteinEncoder` class extending `nn.Module`
  - Load pre-trained ProtBERT or ESM-2 model from Hugging Face
  - Implement `forward()` method that tokenizes sequences and extracts [CLS] token or mean pooling
  - Add batch processing with padding for variable-length sequences
  - _Requirements: 2.1, 2.2, 2.3_

- [ ] 4.2 Add protein encoder configuration
  - Implement option to freeze pre-trained weights or allow fine-tuning
  - Add dropout layer after encoder
  - Handle special tokens and non-standard amino acids
  - _Requirements: 2.4, 2.5_

- [ ]* 4.3 Write protein encoder tests
  - Test output shape is [batch_size, embedding_dim]
  - Test handling of invalid amino acid sequences
  - Test batch processing with variable-length sequences
  - _Requirements: 2.3_

- [ ] 5. Implement co-attention module
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

- [ ] 5.1 Create co-attention mechanism
  - Write `CoAttentionModule` class extending `nn.Module`
  - Implement projection layers for query, key, value transformations
  - Implement scaled dot-product attention: Attention(Q, K, V) = softmax(QK^T / √d_k) V
  - Implement bidirectional attention: drug→protein and protein→drug
  - _Requirements: 3.1, 3.2, 3.4_

- [ ] 5.2 Add attention weight extraction for visualization
  - Store attention weights during forward pass
  - Implement method to return attention matrices [batch, drug_features, protein_features]
  - Add optional attention head averaging for multi-head attention
  - _Requirements: 3.2, 3.5_

- [ ] 5.3 Implement feature fusion
  - Concatenate attended drug and protein representations
  - Add optional fusion strategies (concatenation, addition, gated fusion)
  - Apply layer normalization and dropout
  - _Requirements: 3.3_

- [ ]* 5.4 Write co-attention tests
  - Test attention weight shapes and value ranges (0-1 after softmax)
  - Test bidirectional attention computation
  - Test fusion output shape
  - _Requirements: 3.2, 3.3_

- [ ] 6. Implement evidential regression head
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [ ] 6.1 Create evidential regression layer
  - Write `EvidentialRegressionHead` class extending `nn.Module`
  - Implement MLP that outputs 4 parameters: γ (mean), ν (degrees of freedom), α (shape), β (scale)
  - Add activation functions to ensure positive values for ν, α, β
  - Extract predicted affinity as γ
  - _Requirements: 5.1, 5.2_

- [ ] 6.2 Implement uncertainty calculation
  - Compute epistemic uncertainty from α and β parameters
  - Compute aleatoric uncertainty from ν parameter
  - Compute total uncertainty as combination of both
  - Add method to return uncertainty score normalized to [0, 1] or confidence level (High/Medium/Low)
  - _Requirements: 5.2, 5.4_

- [ ] 6.3 Implement evidential loss function
  - Write `EvidentialLoss` class extending `nn.Module`
  - Implement negative log-likelihood (NLL) term based on Normal-Inverse-Gamma distribution
  - Implement KL divergence regularization term
  - Add hyperparameter λ to balance NLL and KL terms
  - _Requirements: 5.3, 12.1, 12.2_

- [ ]* 6.4 Write evidential regression tests
  - Test output shapes for predictions and uncertainty
  - Test loss computation with sample data
  - Test uncertainty increases for out-of-distribution inputs
  - _Requirements: 5.2, 5.5_

- [ ] 7. Integrate components into complete DTI model
  - _Requirements: 4.1, 4.4, 4.5_

- [ ] 7.1 Create end-to-end DTI model
  - Write `DTIModel` class extending `nn.Module` that integrates DrugEncoder, ProteinEncoder, CoAttentionModule, and EvidentialRegressionHead
  - Implement `forward()` method that chains all components
  - Return dictionary with affinity prediction, uncertainty, and attention weights
  - Add model configuration dataclass
  - _Requirements: 4.1, 4.4_

- [ ] 7.2 Implement model save and load functionality
  - Write methods to save model weights, configuration, and tokenizers
  - Write methods to load model from checkpoint
  - Save optimizer state for training resumption
  - _Requirements: 11.1, 12.5_

- [ ]* 7.3 Write end-to-end model tests
  - Test forward pass with sample SMILES and sequences
  - Test output dictionary contains all required keys
  - Test model save and load preserves weights
  - _Requirements: 4.4, 4.5_

- [ ] 8. Implement training pipeline
  - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5_

- [ ] 8.1 Create training loop
  - Write `Trainer` class with `train_epoch()` method
  - Implement gradient computation and backpropagation with evidential loss
  - Add gradient clipping to prevent exploding gradients
  - Implement learning rate scheduling (e.g., ReduceLROnPlateau or cosine annealing)
  - Add progress logging with tqdm
  - _Requirements: 12.1, 12.2, 12.4_

- [ ] 8.2 Implement validation and metrics
  - Write `validate()` method to evaluate on validation set
  - Compute metrics: MSE, MAE, Pearson correlation, Spearman correlation
  - Evaluate on all three splits: random, cold-drug, cold-protein
  - Log metrics to console and/or TensorBoard
  - _Requirements: 7.3, 12.3_

- [ ] 8.3 Add checkpointing and early stopping
  - Implement model checkpointing based on best validation performance
  - Add early stopping to prevent overfitting
  - Save training history and metrics
  - _Requirements: 12.5_

- [ ] 8.4 Create training script
  - Write `train.py` script that loads data, initializes model, and runs training
  - Add command-line arguments for hyperparameters
  - Add configuration file support (YAML or JSON)
  - _Requirements: 12.3, 12.4_

- [ ]* 8.5 Write training pipeline tests
  - Test training loop runs without errors
  - Test metrics computation accuracy
  - Test checkpointing saves and loads correctly
  - _Requirements: 12.4, 12.5_

- [ ] 9. Implement backend API
  - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_

- [ ] 9.1 Create FastAPI application structure
  - Initialize FastAPI app with CORS middleware
  - Create Pydantic models for request and response schemas
  - Define `PredictionRequest` with smiles and sequence fields
  - Define `PredictionResponse` with affinity, uncertainty, confidence_level, and attention_weights fields
  - _Requirements: 10.1, 10.4, 10.5_

- [ ] 9.2 Implement model service class
  - Write `ModelService` class to load trained model at startup
  - Implement `predict()` method that runs inference
  - Add input validation for SMILES (using RDKit) and protein sequences
  - Handle model loading errors gracefully
  - _Requirements: 10.2, 10.3, 11.1, 11.2_

- [ ] 9.3 Create prediction endpoint
  - Implement POST `/predict` endpoint that accepts PredictionRequest
  - Call ModelService.predict() with input data
  - Format attention weights for frontend consumption
  - Map uncertainty to confidence level (High/Medium/Low based on thresholds)
  - Return PredictionResponse with all required fields
  - _Requirements: 10.1, 10.2, 10.4_

- [ ] 9.4 Add health check and error handling
  - Implement GET `/health` endpoint for service monitoring
  - Add exception handlers for common errors (invalid input, model errors, timeouts)
  - Return appropriate HTTP status codes (400, 500, 503, 504)
  - Add request logging
  - _Requirements: 11.3, 11.4_

- [ ] 9.5 Create backend startup script
  - Write script to start Uvicorn server
  - Add environment variable support for configuration (model path, port, host)
  - Add logging configuration
  - _Requirements: 11.1, 11.5_

- [ ]* 9.6 Write API tests
  - Test prediction endpoint with valid inputs
  - Test error handling for invalid SMILES and sequences
  - Test health check endpoint
  - Test concurrent requests
  - _Requirements: 10.1, 10.4, 11.5_

- [ ] 10. Implement frontend application
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

- [ ] 10.1 Create React application structure
  - Initialize React app with TypeScript using Vite
  - Set up project structure: components/, services/, types/, utils/
  - Configure Axios for API calls with base URL
  - Add Tailwind CSS or Material-UI for styling
  - _Requirements: 8.3_

- [ ] 10.2 Implement input form component
  - Create `InputForm` component with text areas for SMILES and protein sequence
  - Add form validation (non-empty fields)
  - Add submit button with loading state
  - Display validation errors
  - _Requirements: 8.1, 8.2_

- [ ] 10.3 Implement results display component
  - Create `ResultsView` component to display affinity prediction
  - Display uncertainty score and confidence level with visual indicators (colors, badges)
  - Format affinity value with appropriate precision
  - Add clear visual hierarchy
  - _Requirements: 8.4, 8.5_

- [ ] 10.4 Implement attention visualization component
  - Create `AttentionHeatmap` component using D3.js or Plotly
  - Render attention weights as heatmap with SMILES characters on one axis and amino acids on other
  - Add interactive features (hover to see values, zoom)
  - Handle large sequences with scrolling or truncation
  - Add color scale legend
  - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

- [ ] 10.5 Integrate components and API calls
  - Create main `App` component that manages state
  - Implement API call to `/predict` endpoint on form submission
  - Handle loading, success, and error states
  - Pass prediction results to ResultsView and AttentionHeatmap
  - Add error display for API failures
  - _Requirements: 8.3, 8.4, 8.5_

- [ ] 10.6 Add example inputs and documentation
  - Provide example SMILES and protein sequences for users to try
  - Add tooltips or help text explaining inputs
  - Create simple user guide or FAQ section
  - _Requirements: 8.1, 8.2_

- [ ]* 10.7 Write frontend tests
  - Test InputForm validation and submission
  - Test ResultsView displays data correctly
  - Test AttentionHeatmap renders without errors
  - Test error handling for API failures
  - _Requirements: 8.3, 8.4, 8.5_

- [ ] 11. Create deployment configuration
  - _Requirements: 11.1, 11.2, 11.3_

- [ ] 11.1 Write Dockerfiles
  - Create Dockerfile for backend with Python dependencies
  - Create Dockerfile for frontend with Node.js build
  - Optimize image sizes with multi-stage builds
  - _Requirements: 11.1_

- [ ] 11.2 Create Docker Compose configuration
  - Write docker-compose.yml to orchestrate backend and frontend services
  - Configure port mappings (8000 for backend, 3000 for frontend)
  - Set up volume mounts for model weights
  - Add environment variables for configuration
  - _Requirements: 11.1, 11.2_

- [ ] 11.3 Write deployment documentation
  - Create README with setup instructions
  - Document environment variables and configuration options
  - Add troubleshooting guide
  - Include example commands for running locally and with Docker
  - _Requirements: 11.1, 11.3_

- [ ] 12. End-to-end integration and validation
  - _Requirements: 7.1, 7.2, 7.3, 7.5_

- [ ] 12.1 Run full training pipeline
  - Execute data preprocessing on BindingDB dataset
  - Train model on random split for baseline performance
  - Evaluate on all three validation splits
  - Save best model checkpoint
  - _Requirements: 12.3, 12.4, 12.5_

- [ ] 12.2 Validate model performance
  - Compute and report metrics on random, cold-drug, and cold-protein test sets
  - Analyze uncertainty calibration
  - Generate sample attention visualizations for qualitative assessment
  - Document performance results
  - _Requirements: 7.1, 7.2, 7.3, 7.5_

- [ ] 12.3 Test full-stack integration
  - Start backend API with trained model
  - Start frontend application
  - Test end-to-end prediction flow with sample inputs
  - Verify attention visualization displays correctly
  - Test error handling with invalid inputs
  - _Requirements: 8.3, 8.4, 8.5, 10.1, 10.4, 11.3_

- [ ] 12.4 Deploy with Docker
  - Build Docker images for backend and frontend
  - Run docker-compose up to start all services
  - Test deployed application
  - Verify model loading and inference work correctly
  - _Requirements: 11.1, 11.2_

- [ ]* 12.5 Performance optimization
  - Profile inference latency and identify bottlenecks
  - Optimize model loading time
  - Test concurrent request handling
  - Implement caching if needed
  - _Requirements: 11.3, 11.5_
