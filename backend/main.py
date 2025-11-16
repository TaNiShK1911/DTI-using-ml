from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import sys
import os
from typing import List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.dti_model import DTIModel
from rdkit import Chem
from rdkit.Chem import Descriptors

app = FastAPI(title="DTI Prediction API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    smiles: str
    sequence: str


class PredictionResponse(BaseModel):
    affinity: float
    uncertainty: float
    confidence_level: str
    attention_weights: List[List[float]]
    detailed_explanation: dict


class ModelService:
    """Service for model inference."""
    
    def __init__(self, model_path: str = 'checkpoints/best_model.pt'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_loaded = False
        
        print(f"Initializing model service on {self.device}...")
        
        # Always try to initialize with untrained model first as fallback
        try:
            self.model = DTIModel()
            self.model.to(self.device)
            self.model.eval()
            print("✓ Base model initialized")
        except Exception as e:
            print(f"✗ Critical error: Cannot initialize model: {e}")
            raise
        
        # Try to load checkpoint if it exists
        if not os.path.exists(model_path):
            print(f"⚠ Checkpoint not found at {model_path}")
            print("  Using untrained model")
            return
        
        try:
            print(f"Loading checkpoint from {model_path}...")
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Load state dict with strict=False to handle mismatches
            missing_keys, unexpected_keys = self.model.load_state_dict(
                checkpoint['model_state_dict'], 
                strict=False
            )
            
            if not missing_keys and not unexpected_keys:
                print("✓ Model loaded successfully with all weights!")
                self.model_loaded = True
            else:
                print("⚠ Model loaded with some mismatches:")
                if missing_keys:
                    print(f"  Missing keys: {len(missing_keys)}")
                if unexpected_keys:
                    print(f"  Unexpected keys: {len(unexpected_keys)}")
                print("  Using partially loaded model")
                self.model_loaded = True
            
        except Exception as e:
            print(f"⚠ Could not load checkpoint: {e}")
            print("  Continuing with untrained model")
    
    def validate_smiles(self, smiles: str) -> bool:
        """Validate SMILES string."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False
    
    def validate_sequence(self, sequence: str) -> bool:
        """Validate protein sequence."""
        valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
        return len(sequence) > 0 and all(c in valid_aa for c in sequence.upper())
    
    def predict(self, smiles: str, sequence: str) -> dict:
        """Run inference."""
        # Validate inputs
        if not self.validate_smiles(smiles):
            raise ValueError("Invalid SMILES string")
        
        if not self.validate_sequence(sequence):
            raise ValueError("Invalid protein sequence")
        
        # Run inference
        with torch.no_grad():
            outputs = self.model([smiles], [sequence.upper()])
        
        affinity = outputs['affinity'][0].item()
        uncertainty = outputs['uncertainty'][0].item()
        attention_weights = outputs['attention_weights'][0].cpu().numpy().tolist()
        evidential_params = outputs['evidential_params'][0].cpu().numpy()
        
        # Extract evidential parameters
        gamma = evidential_params[0]  # Mean (affinity prediction)
        nu = evidential_params[1]     # Degrees of freedom
        alpha = evidential_params[2]  # Shape parameter
        beta = evidential_params[3]   # Scale parameter
        
        # Calculate uncertainty components
        epistemic = beta / (alpha - 1 + 1e-8)
        aleatoric = beta / (nu * (alpha - 1) + 1e-8)
        total_uncertainty = epistemic + aleatoric
        
        # Map uncertainty to confidence level with thresholds
        if uncertainty < 0.3:
            confidence_level = "High"
            confidence_description = "Model is very confident in this prediction"
        elif uncertainty < 0.8:
            confidence_level = "Medium"
            confidence_description = "Model has moderate confidence in this prediction"
        else:
            confidence_level = "Low"
            confidence_description = "Model has low confidence - prediction may be unreliable"
        
        # Create detailed explanation
        detailed_explanation = {
            "affinity_calculation": {
                "raw_prediction": float(gamma),
                "final_affinity": float(affinity),
                "interpretation": f"Predicted binding affinity of {affinity:.2f} on pKi/pKd/pIC50 scale",
                "binding_strength": self._interpret_affinity(affinity)
            },
            "uncertainty_breakdown": {
                "total_uncertainty": float(uncertainty),
                "epistemic_uncertainty": float(epistemic),
                "aleatoric_uncertainty": float(aleatoric),
                "confidence_level": confidence_level,
                "confidence_description": confidence_description,
                "explanation": {
                    "epistemic": "Model uncertainty - how uncertain the model is about its parameters",
                    "aleatoric": "Data uncertainty - inherent noise in the data",
                    "total": "Combined uncertainty score"
                }
            },
            "evidential_parameters": {
                "gamma": float(gamma),
                "nu": float(nu),
                "alpha": float(alpha),
                "beta": float(beta),
                "parameter_meanings": {
                    "gamma": "Mean prediction (affinity value)",
                    "nu": "Degrees of freedom (higher = more certain about noise level)",
                    "alpha": "Shape parameter (higher = more certain about mean)",
                    "beta": "Scale parameter (affects uncertainty magnitude)"
                }
            },
            "attention_analysis": {
                "attention_shape": f"{len(attention_weights)} x {len(attention_weights[0]) if attention_weights else 0}",
                "max_attention": float(max(max(row) for row in attention_weights)) if attention_weights else 0.0,
                "min_attention": float(min(min(row) for row in attention_weights)) if attention_weights else 0.0,
                "interpretation": "Higher values indicate stronger predicted interactions between drug and protein features"
            },
            "input_analysis": {
                "smiles_length": len(smiles),
                "sequence_length": len(sequence),
                "smiles_complexity": self._analyze_smiles_complexity(smiles),
                "sequence_composition": self._analyze_sequence_composition(sequence)
            }
        }
        
        return {
            'affinity': affinity,
            'uncertainty': uncertainty,
            'confidence_level': confidence_level,
            'attention_weights': attention_weights,
            'detailed_explanation': detailed_explanation
        }
    
    def _interpret_affinity(self, affinity: float) -> str:
        """Interpret affinity value in biological terms."""
        if affinity >= 9.0:
            return "Very strong binding (sub-nanomolar)"
        elif affinity >= 7.0:
            return "Strong binding (nanomolar range)"
        elif affinity >= 5.0:
            return "Moderate binding (micromolar range)"
        elif affinity >= 3.0:
            return "Weak binding (high micromolar)"
        else:
            return "Very weak or no significant binding"
    
    def _analyze_smiles_complexity(self, smiles: str) -> dict:
        """Analyze SMILES complexity."""
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                return {
                    "num_atoms": mol.GetNumAtoms(),
                    "num_bonds": mol.GetNumBonds(),
                    "num_rings": mol.GetRingInfo().NumRings(),
                    "molecular_weight": Chem.Descriptors.MolWt(mol)
                }
        except:
            pass
        return {
            "num_atoms": "N/A",
            "num_bonds": "N/A", 
            "num_rings": "N/A",
            "molecular_weight": "N/A"
        }
    
    def _analyze_sequence_composition(self, sequence: str) -> dict:
        """Analyze protein sequence composition."""
        aa_counts = {}
        for aa in sequence:
            aa_counts[aa] = aa_counts.get(aa, 0) + 1
        
        return {
            "length": len(sequence),
            "unique_amino_acids": len(aa_counts),
            "most_common": max(aa_counts.items(), key=lambda x: x[1]) if aa_counts else ("N/A", 0),
            "composition": aa_counts
        }


# Initialize model service
model_service = ModelService()


@app.get("/")
async def root():
    return {"message": "DTI Prediction API", "status": "running"}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "device": model_service.device,
        "model_loaded": model_service.model_loaded,
        "debug_mode": os.environ.get('SKIP_MODEL_LOAD') == '1'
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_binding(request: PredictionRequest):
    """Main prediction endpoint."""
    try:
        result = model_service.predict(request.smiles, request.sequence)
        return PredictionResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
