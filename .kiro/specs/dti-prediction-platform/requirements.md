# Requirements Document

## Introduction

This document specifies the requirements for a novel Drug-Target Interaction (DTI) prediction platform. The DTI Platform is a full-stack web application that predicts binding affinity between drug molecules and protein targets using a multimodal deep learning model. The system integrates advanced features including uncertainty quantification and interpretable attention mechanisms to provide transparent, confidence-scored predictions suitable for pharmaceutical research and drug discovery workflows.

## Glossary

- **DTI Platform**: The complete Drug-Target Interaction prediction web application system
- **Binding Affinity Predictor**: The deep learning model component that predicts interaction strength between drugs and proteins
- **SMILES Encoder**: The neural network component that processes SMILES string representations of drug molecules
- **Protein Encoder**: The neural network component that processes amino acid sequences of target proteins
- **Co-Attention Module**: The neural network component that computes cross-modal attention weights between drug and protein representations
- **Uncertainty Estimator**: The model component that quantifies prediction confidence using evidential deep learning
- **Frontend Interface**: The React-based user interface for input and visualization
- **Backend API**: The Flask or FastAPI service that handles inference requests
- **BindingDB Dataset**: The source TSV file containing drug-protein binding measurements
- **Cold-Start Validation**: The evaluation strategy using test sets with unseen drugs or proteins
- **Attention Visualization**: The interactive display showing predicted binding hotspots

## Requirements

### Requirement 1: Multimodal Drug Representation

**User Story:** As a computational chemist, I want the system to process drug molecules using both SMILES strings and molecular graph representations, so that the model captures comprehensive structural information for accurate binding predictions.

#### Acceptance Criteria

1. WHEN a SMILES string is provided as input, THE SMILES Encoder SHALL generate a fixed-dimensional embedding vector using a pre-trained transformer model or Graph Neural Network
2. THE SMILES Encoder SHALL utilize either MolBERT or an equivalent pre-trained SMILES-based transformer architecture
3. THE SMILES Encoder SHALL accept SMILES strings of variable length and output embeddings of consistent dimensionality
4. WHERE a Graph Neural Network is used, THE SMILES Encoder SHALL convert SMILES strings to 2D molecular graph representations before encoding
5. THE SMILES Encoder SHALL preserve molecular structural features relevant to protein binding interactions

### Requirement 2: Protein Sequence Encoding

**User Story:** As a structural biologist, I want the system to encode protein sequences using state-of-the-art language models, so that the model understands the biological context and functional motifs of target proteins.

#### Acceptance Criteria

1. WHEN an amino acid sequence is provided as input, THE Protein Encoder SHALL generate a fixed-dimensional embedding vector using a pre-trained protein language model
2. THE Protein Encoder SHALL utilize either ProtBERT or ESM as the pre-trained protein language model
3. THE Protein Encoder SHALL accept amino acid sequences of variable length and output embeddings of consistent dimensionality
4. THE Protein Encoder SHALL capture sequence-level features including secondary structure and functional motifs
5. WHERE protein targets contain multiple chains, THE Protein Encoder SHALL process the first chain sequence from the BindingDB Dataset

### Requirement 3: Multimodal Fusion with Co-Attention

**User Story:** As a machine learning researcher, I want the model to learn cross-modal interactions between drug and protein representations, so that the system can identify which molecular substructures interact with specific protein regions.

#### Acceptance Criteria

1. THE Co-Attention Module SHALL compute attention weights between drug embeddings and protein embeddings
2. THE Co-Attention Module SHALL output attention matrices that quantify interaction strength between drug features and protein features
3. THE Binding Affinity Predictor SHALL fuse the attended drug and protein representations before generating predictions
4. THE Co-Attention Module SHALL enable bidirectional attention flow from drug to protein and protein to drug
5. THE Co-Attention Module SHALL produce interpretable attention weights suitable for visualization

### Requirement 4: Binding Affinity Prediction

**User Story:** As a medicinal chemist, I want the system to predict quantitative binding affinity values, so that I can rank and prioritize drug candidates for experimental validation.

#### Acceptance Criteria

1. THE Binding Affinity Predictor SHALL output a numerical binding affinity prediction for each drug-protein pair
2. THE Binding Affinity Predictor SHALL predict affinity values in standardized units derived from Ki, Kd, or IC50 measurements
3. THE Binding Affinity Predictor SHALL apply p-scaling transformation where pX equals negative log base 10 of X nanomolar
4. THE Binding Affinity Predictor SHALL accept fused multimodal representations from the Co-Attention Module as input
5. THE Binding Affinity Predictor SHALL produce predictions within a biologically plausible range based on training data distribution

### Requirement 5: Uncertainty Quantification

**User Story:** As a pharmaceutical researcher, I want the system to provide confidence scores with each prediction, so that I can assess the reliability of predictions and make informed decisions about experimental follow-up.

#### Acceptance Criteria

1. THE Uncertainty Estimator SHALL output a confidence score or uncertainty distribution for each binding affinity prediction
2. THE Uncertainty Estimator SHALL implement evidential deep learning or an equivalent Bayesian uncertainty quantification method
3. THE Binding Affinity Predictor SHALL incorporate an evidential loss function during training
4. THE Uncertainty Estimator SHALL distinguish between aleatoric uncertainty and epistemic uncertainty in predictions
5. THE Uncertainty Estimator SHALL provide higher uncertainty scores for inputs that differ significantly from training data distribution

### Requirement 6: Data Preprocessing from BindingDB

**User Story:** As a data scientist, I want the system to automatically extract and preprocess relevant data from the BindingDB TSV file, so that the model trains on clean, standardized binding affinity measurements.

#### Acceptance Criteria

1. WHEN processing the BindingDB Dataset, THE DTI Platform SHALL extract Ligand SMILES strings from the designated column
2. WHEN processing the BindingDB Dataset, THE DTI Platform SHALL extract target protein sequences from the BindingDB Target Chain Sequence 1 column
3. THE DTI Platform SHALL create a unified target affinity column by prioritizing Ki over Kd over IC50 measurements
4. THE DTI Platform SHALL filter out all rows where the selected affinity measurement is blank or unavailable
5. THE DTI Platform SHALL apply p-scaling transformation to convert nanomolar affinity values to standardized pX values

### Requirement 7: Cold-Start Validation Strategy

**User Story:** As a machine learning scientist, I want the model to be evaluated on unseen drugs and proteins separately, so that I can demonstrate the model's generalization capability and robustness for real-world drug discovery scenarios.

#### Acceptance Criteria

1. THE DTI Platform SHALL implement a cold-start validation split where the test set contains only drugs unseen during training
2. THE DTI Platform SHALL implement a cold-start validation split where the test set contains only proteins unseen during training
3. THE DTI Platform SHALL report performance metrics separately for the unseen-drug split and the unseen-protein split
4. THE DTI Platform SHALL ensure zero overlap between training and test sets for the respective cold-start conditions
5. THE DTI Platform SHALL use cold-start validation results to assess model generalization beyond random split performance

### Requirement 8: Frontend User Interface

**User Story:** As an end user, I want a simple web interface to input drug and protein data and view prediction results, so that I can quickly assess binding potential without requiring programming knowledge.

#### Acceptance Criteria

1. THE Frontend Interface SHALL provide an input field for users to enter a drug molecule as a SMILES string
2. THE Frontend Interface SHALL provide an input field for users to enter a target protein as an amino acid sequence
3. THE Frontend Interface SHALL be implemented using React or Next.js framework
4. WHEN a user submits valid inputs, THE Frontend Interface SHALL display the predicted binding affinity value
5. THE Frontend Interface SHALL display the uncertainty or confidence level associated with each prediction

### Requirement 9: Interpretable Attention Visualization

**User Story:** As a researcher, I want to see which parts of the drug molecule interact with which regions of the protein, so that I can understand the molecular basis of the predicted binding and guide rational drug design.

#### Acceptance Criteria

1. THE Attention Visualization SHALL display attention weights computed by the Co-Attention Module
2. THE Attention Visualization SHALL highlight predicted binding hotspots on both the drug structure and protein sequence
3. THE Frontend Interface SHALL render the Attention Visualization in an interactive and interpretable format
4. THE Attention Visualization SHALL use visual encoding such as heatmaps or highlighting to represent attention strength
5. THE Attention Visualization SHALL update dynamically based on the current drug-protein pair prediction

### Requirement 10: Backend API for Inference

**User Story:** As a frontend developer, I want a RESTful API endpoint that accepts drug and protein inputs and returns predictions, so that the frontend can communicate with the model seamlessly.

#### Acceptance Criteria

1. THE Backend API SHALL expose an endpoint that accepts SMILES strings and amino acid sequences as input parameters
2. WHEN the Backend API receives a request, THE Backend API SHALL preprocess inputs by generating embeddings using the SMILES Encoder and Protein Encoder
3. THE Backend API SHALL load the trained Binding Affinity Predictor model weights for inference
4. THE Backend API SHALL return a JSON response containing the binding affinity prediction, uncertainty score, and attention weights
5. THE Backend API SHALL be implemented using Flask or FastAPI framework

### Requirement 11: Model Deployment and Inference

**User Story:** As a DevOps engineer, I want the trained model to be deployable as a service that can handle real-time inference requests, so that users can interact with the platform without delays.

#### Acceptance Criteria

1. THE Backend API SHALL load saved PyTorch or TensorFlow model weights at service initialization
2. THE Backend API SHALL perform inference on new drug-protein pairs in real-time upon receiving API requests
3. THE Backend API SHALL complete inference and return results within a reasonable response time for interactive use
4. THE DTI Platform SHALL handle model loading errors gracefully and provide informative error messages
5. THE Backend API SHALL support concurrent inference requests from multiple users

### Requirement 12: Model Training with Evidential Loss

**User Story:** As a machine learning engineer, I want the model training process to incorporate uncertainty quantification from the start, so that the model learns to estimate its own confidence alongside making predictions.

#### Acceptance Criteria

1. WHEN training the Binding Affinity Predictor, THE DTI Platform SHALL use an evidential loss function
2. THE DTI Platform SHALL optimize both prediction accuracy and uncertainty calibration during training
3. THE DTI Platform SHALL train the model on the preprocessed BindingDB Dataset with unified affinity targets
4. THE DTI Platform SHALL implement gradient-based optimization for all model components including the SMILES Encoder, Protein Encoder, Co-Attention Module, and prediction heads
5. THE DTI Platform SHALL save model checkpoints and final weights for deployment to the Backend API
